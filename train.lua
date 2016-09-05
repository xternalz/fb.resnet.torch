--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
   self.opt = opt
   --self.params, self.gradParams = model:getParameters()
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)

   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = dataloader:size()
   local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
   local N = 0

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1)
      local loss = self.criterion:forward(self.model.output, self.target)

      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)

      optim.sgd(feval, self.params, self.optimState)

      local top1, top5 = self:computeScore(output, sample.target, 1)
      top1Sum = top1Sum + top1*batchSize
      top5Sum = top5Sum + top5*batchSize
      lossSum = lossSum + loss*batchSize
      N = N + batchSize

      if n % 100 == 0 or n == 1 then
         print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f  top1 %7.3f  top5 %7.3f'):format(
            epoch, n, trainSize, timer:time().real, dataTime, loss, top1, top5))
      end

      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
   end

   return top1Sum / N, top5Sum / N, lossSum / N
end

function Trainer:test(epoch, dataloader, scale)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = 10

   --Find out if there's any existing results saved
   local saveInd = 1
   local indices = torch.ones(dataloader.__size)
   if not paths.dirp('results') then
      paths.mkdir('results')
   end
   if not paths.dirp('results/' .. scale) then
      paths.mkdir('results/' .. scale)
   end
   for filename in paths.iterfiles('results/' .. scale) do
      if string.match(filename, '.t7') then
         local res = torch.load('results/' .. scale .. '/'.. filename)
         res = res[3]
         for j = 1, res:size(1) do
            if res[j] ~= -1 then
               indices[res[j]] = 0
            end
         end
         saveInd = saveInd + 1
      end
   end
   indices = indices:nonzero():squeeze()
   if indices:nElement() == 0 then
      return 0, 0
   end
   collectgarbage()

   local dataloaderBatchSize = dataloader.batchSize
   local resultCount = 1
   local results = {torch.FloatTensor(2,self.opt.saveInterval*dataloaderBatchSize,365), torch.LongTensor(self.opt.saveInterval*dataloaderBatchSize):fill(-1)}

   self.model:evaluate()
   local softmax = cudnn.SoftMax():cuda()
   for n, sample in dataloader:run(indices, scale) do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      -- Stochastic inference
      local raw_output = nil
      local prob_output = nil
      for i = 1, self.opt.nStocSamples do
         local output = nil
         if self.opt.frontModelDet == false then
            output = self.model:forward(self.input)
         else
            if i == 1 then
               output = self.model:get(2):forward(self.model:get(1):forward(self.input))
            else
               output = self.model:get(2):forward(self.model:get(1).output)
            end
         end
         if raw_output == nil then
            raw_output = output:float():clone()
         else
            raw_output:add(output:float())
         end
         if prob_output == nil then
            prob_output = softmax:forward(output):float():clone()
         else
            prob_output:add(softmax:forward(output):float())
         end
      end
      raw_output:div(self.opt.nStocSamples)
      prob_output:div(self.opt.nStocSamples)

      local batchCount = self.index:size(1)
      results[1]:select(1,1):narrow(1,resultCount,batchCount):copy(raw_output:view(raw_output:size(1)/6,6,raw_output:size(2)):mean(2):squeeze(2))
      results[1]:select(1,2):narrow(1,resultCount,batchCount):copy(prob_output:view(prob_output:size(1)/6,6,prob_output:size(2)):mean(2):squeeze(2))
      results[2]:narrow(1,resultCount,batchCount):copy(self.index)
      resultCount = resultCount + batchCount

      if resultCount >= self.opt.saveInterval*dataloaderBatchSize or n == size then
         torch.save('results/' .. scale .. '/' .. saveInd .. '.t7', results)
         saveInd = saveInd + 1
         resultCount = 1
         results[2]:fill(-1)
      end

      print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f'):format(
         epoch, n, size, timer:time().real, dataTime))

      timer:reset()
      dataTimer:reset()
   end

   return 0, 0
end

function Trainer:computeScore(output, target, nCrops, top5ClassAccs)
   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():sort(2, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(output))

   -- Top-1 score
   local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

   -- Top-5 score, if there are at least 5 classes
   local len = math.min(5, correct:size(2))
   local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

   -- Per-class Top-5
   local correct_ = correct:narrow(2, 1, len):sum(2):squeeze()
   local target_ = target:long()
   for i = 1, batchSize do
      top5ClassAccs[target_[i]][2] = top5ClassAccs[target_[i]][2] + 1
      if correct_[i] > 0 then
         top5ClassAccs[target_[i]][1] = top5ClassAccs[target_[i]][1] + 1
      end
   end

   return top1 * 100, top5 * 100
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.target = self.target or torch.CudaTensor()
   self.index = self.index or torch.LongTensor()

   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
   self.index:resize(sample.index:size()):copy(sample.index)
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if string.match(self.opt.dataset, 'imagenet') then
      decay = math.floor((epoch - 1) / 30)
   elseif string.match(self.opt.dataset, 'cifar10') then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   end
   return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer
