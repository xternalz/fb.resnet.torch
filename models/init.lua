--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Generic model creating code. For the specific ResNet model see
--  models/resnet.lua
--

require 'nn'
require 'cunn'
require 'cudnn'

local M = {}

function M.setup(opt, checkpoint)
   local model
   if checkpoint then
      local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
      assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
      print('=> Resuming model from ' .. modelPath)
      model = torch.load(modelPath):cuda()
   elseif opt.retrain ~= 'none' then
      assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
      print('Loading model from file: ' .. opt.retrain)
      model = torch.load(opt.retrain):cuda()
   else
      print('=> Creating model from file: models/' .. opt.netType .. '.lua')
      model = require('models/' .. opt.netType)(opt)
   end

   local model_classify = model
   if model:size() == 2 then
      model_classify = model_classify:get(2)
   end
   --[[local poolInd = model_classify:size()-2
   model_classify:remove(poolInd)
   model_classify:insert(nn.Squeeze(2,2), poolInd)
   model_classify:insert(nn.Mean(3), poolInd)
   model_classify:insert(nn.Squeeze(3,3), poolInd)
   model_classify:insert(nn.View(model_classify:get(model_classify:size()).weight:size(2),-1,1):setNumInputDims(3), poolInd)--]]
   model_classify:remove(model_classify:size()-1)

   -- convert FC to conv layer
   local fc = model_classify:get(model_classify:size())
   model_classify:insert(nn.SpatialConvolution(fc.weight:size(2),fc.weight:size(1),1,1,0,0),model_classify:size())
   model_classify:get(model_classify:size()-1).weight:select(3,1):select(3,1):copy(model_classify:get(model_classify:size()).weight)
   model_classify:get(model_classify:size()-1).bias:copy(model_classify:get(model_classify:size()).bias)
   model_classify:remove(model_classify:size())

   model:cuda()
   local optnet = require 'optnet'
   local sampleInput = torch.zeros(4,3,320,320):cuda()
   model:evaluate()
   model:apply(function(m)
      if torch.type(m) == 'nn.Dropout' then
         m.train = true
      end
   end)
   if opt.frontModelDet == false then
      optnet.optimizeMemory(model, sampleInput, {inplace = true, mode = 'inference', reuseBuffers = true, removeGradParams = true})
   else
      optnet.optimizeMemory(model:get(1), sampleInput, {inplace = true, mode = 'inference', reuseBuffers = true, removeGradParams = true})
      local sampleInput2 = model:get(1):forward(sampleInput)
      optnet.optimizeMemory(model:get(2), sampleInput2, {inplace = true, mode = 'inference', reuseBuffers = true, removeGradParams = true})
   end
   model:evaluate()
   model:apply(function(m)
      if torch.type(m) == 'nn.Dropout' then
         m.train = true
      end
   end)
   M.shareDropoutNoise(model)

   -- Add front resnet to model
   model.backward = function() end
   model.updateGradInput = function() end
   model.accGradParameters = function() end
   model.accUpdateGradParameters = function() end
   model.accUpdateGradParameters = function() end
   model.training = function() end
   model.evaluate = function() end
   local a = torch.CudaTensor(1)
   model.parameters = function() return {a}, {a} end
   if opt.frontModelDet == true then
      for i = 1, 2 do
         model:get(i).backward = function() end
         model:get(i).updateGradInput = function() end
         model:get(i).accGradParameters = function() end
         model:get(i).accUpdateGradParameters = function() end
         model:get(i).accUpdateGradParameters = function() end
         model:get(i).training = function() end
         model:get(i).evaluate = function() end
         local a = torch.CudaTensor(1)
         model:get(i).parameters = function() return {a}, {a} end
      end
   end

   -- First remove any DataParallelTable
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   -- optnet is an general library for reducing memory usage in neural networks
   if opt.optnet then
      local optnet = require 'optnet'
      local imsize = string.match(opt.dataset, 'imagenet') ~= nil and 224 or 32
      local sampleInput = torch.zeros(4,3,imsize,imsize):cuda()
      optnet.optimizeMemory(model, sampleInput, {inplace = false, mode = 'training'})
   end

   -- This is useful for fitting ResNet-50 on 4 GPUs, but requires that all
   -- containers override backwards to call backwards recursively on submodules
   if opt.shareGradInput then
      M.shareGradInput(model)
   end

   -- For resetting the classifier when fine-tuning on a different Dataset
   if opt.resetClassifier and not checkpoint then
      print(' => Replacing classifier with ' .. opt.nClasses .. '-way classifier')

      local orig = model:get(#model.modules)
      assert(torch.type(orig) == 'nn.Linear',
         'expected last layer to be fully connected')

      local linear = nn.Linear(orig.weight:size(2), opt.nClasses)
      linear.bias:zero()

      model:remove(#model.modules)
      model:add(linear:cuda())
   end

   -- Set the CUDNN flags
   if opt.cudnn == 'fastest' then
      cudnn.fastest = true
      cudnn.benchmark = true
   elseif opt.cudnn == 'deterministic' then
      -- Use a deterministic convolution implementation
      model:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
   end

   -- Wrap the model with DataParallelTable, if using more than one GPU
   if opt.nGPU > 1 then
      local gpus = torch.range(1, opt.nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      if opt.frontModelDet == false then
         local dpt = nn.DataParallelTable(1, true, true)
            :add(model, gpus)
            :threads(function()
               local cudnn = require 'cudnn'
               cudnn.fastest, cudnn.benchmark = fastest, benchmark
            end)
         dpt.gradInput = nil
         model = dpt:cuda()
      else
         local models = nn.Sequential()
         for i = 1, 2 do
            local dpt = nn.DataParallelTable(1, true, true)
               :add(model:get(i), gpus)
               :threads(function()
                  local cudnn = require 'cudnn'
                  cudnn.fastest, cudnn.benchmark = fastest, benchmark
               end)
            dpt.gradInput = nil
            models:add(dpt:cuda())
         end
         model = models
      end
   end

   local criterion = nn.CrossEntropyCriterion():cuda()
   return model, criterion
end

function M.shareGradInput(model)
   local function sharingKey(m)
      local key = torch.type(m)
      if m.__shareGradInputKey then
         key = key .. ':' .. m.__shareGradInputKey
      end
      return key
   end

   -- Share gradInput for memory efficient backprop
   local cache = {}
   model:apply(function(m)
      local moduleType = torch.type(m)
      if torch.isTensor(m.gradInput) and moduleType ~= 'nn.ConcatTable' then
         local key = sharingKey(m)
         if cache[key] == nil then
            cache[key] = torch.CudaStorage(1)
         end
         m.gradInput = torch.CudaTensor(cache[key], 1, 0)
      end
   end)
   for i, m in ipairs(model:findModules('nn.ConcatTable')) do
      if cache[i % 2] == nil then
         cache[i % 2] = torch.CudaStorage(1)
      end
      m.gradInput = torch.CudaTensor(cache[i % 2], 1, 0)
   end
end

function M.shareDropoutNoise(model)
   local function sharingKey(m)
      local key = torch.type(m)
      if m.__shareDropoutNoise then
         key = key .. ':' .. m.__shareDropoutNoise
      end
      return key
   end

   -- Share dropoutNoise for memory efficient forward
   local cache = {}
   model:apply(function(m)
      local moduleType = torch.type(m)
      if torch.isTensor(m.noise) then
         local key = sharingKey(m)
         if cache[key] == nil then
            cache[key] = torch.CudaStorage(1)
         end
         m.noise = torch.CudaTensor(cache[key], 1, 0)
      end
   end)
end

return M
