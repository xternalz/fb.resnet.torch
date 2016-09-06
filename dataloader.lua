--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('resnet.DataLoader', M)

function DataLoader.create(opt)
   -- The train and val loader
   local loaders = {}

   for i, split in ipairs{'train', 'val'} do
      --local dataset = datasets.create(opt, split)
      local imageInfo = datasets.getImageInfo(opt, split)
      loaders[i] = M.DataLoader(imageInfo, opt, split)
   end

   return table.unpack(loaders)
end

function DataLoader:__init(imageInfo, opt, split)
   local manualSeed = opt.manualSeed
   local function init()
      --require('datasets/' .. opt.dataset)
   end
   local function main(idx)
      if manualSeed ~= 0 then
         torch.manualSeed(manualSeed + idx)
      end
      torch.setnumthreads(1)
      local dataset = require('datasets/' .. opt.dataset)
      _G.dataset = dataset(imageInfo, opt, split)
      _G.preprocess = _G.dataset:preprocess()
      return _G.dataset:size()
   end

   local threads, sizes = nil, nil
   if split == 'train' then
      threads, sizes = Threads(1, init, main)
   else
      threads, sizes = Threads(opt.nThreads, init, main)
   end
   self.nCrops = opt.nCrop
   self.threads = threads
   self.__size = sizes[1][1]
   self.__size = self.__size
   self.batchSize = math.floor(opt.batchSize / self.nCrops)
   self.nIters = math.ceil(self.__size / self.batchSize)
   self.split = split

   -- organize training samples by classes
   if split == 'train' then
      local labels = imageInfo[split].imageClass
      self.classKeyInds = {}
      self.classInds = {}
      self.classIndPointer = {}
      for i = 1, #labels do
         local label = labels[i]
         if self.classKeyInds[label] == nil then
            self.classKeyInds[label] = {i}
         else
            table.insert(self.classKeyInds[label], i)
         end
         if self.classInds[label] == nil then
            self.classInds[label] = 1
            self.classIndPointer[label] = 1
         else
            self.classInds[label] = self.classInds[label] + 1
         end
         if i % 10000 == 0 then
            collectgarbage()
         end
      end
      for i = 1, #self.classInds do
         local numThisClass = self.classInds[i]
         self.classInds[i] = torch.randperm(numThisClass)
      end
      self.classes = torch.randperm(#self.classInds)
      self.classPointer = 1
      collectgarbage()
   end
end

function DataLoader:size()
   return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run(indices_to_process, input_scale)
   local threads = self.threads
   local size, batchSize = self.__size, self.batchSize
   local perm = indices_to_process
   size = indices_to_process:size(1)

   local idi, idx, sample = 0, 1, nil
   local function enqueue()
      while ((idx <= size and self.split == 'val') or (idi < self.nIters and self.split == 'train')) and threads:acceptsjob() do
         local indices
         if self.split == 'val' then
            indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
         elseif self.split == 'train' then
            local indicesCatTable = {}
            for k = 1, batchSize do
               if self.classPointer > self.classes:size(1) then
                  self.classes = torch.randperm(self.classes:size(1))
                  self.classPointer = 1
               end
               local class = self.classes[self.classPointer]
               self.classPointer = self.classPointer + 1
               if self.classIndPointer[class] > self.classInds[class]:size(1) then
                  self.classInds[class] = torch.randperm(self.classInds[class]:size(1))
                  self.classIndPointer[class] = 1
               end
               table.insert(indicesCatTable, self.classKeyInds[class][self.classInds[class][self.classIndPointer[class]]])
               self.classIndPointer[class] = self.classIndPointer[class] + 1
            end
            indices = torch.Tensor(indicesCatTable)
            collectgarbage()
         end
         threads:addjob(
            function(indices, nCrops, scale)
               local sz = indices:size(1)
               local batch, imageSize
               local target = torch.IntTensor(sz)
               for i, idx in ipairs(indices:totable()) do
                  local sample = _G.dataset:get(idx)
                  local input = _G.preprocess(sample.input, scale)
                  if not batch then
                     imageSize = input:size():totable()
                     if nCrops > 1 then table.remove(imageSize, 1) end
                     batch = torch.FloatTensor(sz, nCrops, table.unpack(imageSize))
                  end
                  batch[i]:copy(input)
                  target[i] = sample.target
               end
               collectgarbage()
               return {
                  input = batch:view(sz * nCrops, table.unpack(imageSize)),
                  target = target,
                  index = indices
               }
            end,
            function(_sample_)
               sample = _sample_
            end,
            indices,
            self.nCrops,
            input_scale
         )
         idx = idx + batchSize
         idi = idi + 1
      end
   end

   local n = 0
   local function loop()
      enqueue()
      if not threads:hasjob() then
         return nil
      end
      threads:dojob()
      if threads:haserror() then
         threads:synchronize()
      end
      enqueue()
      n = n + 1
      return n, sample
   end

   return loop
end

return M.DataLoader
