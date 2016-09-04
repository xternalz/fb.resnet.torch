--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script to compute list of ImageNet filenames and classes
--
--  This generates a file gen/imagenet.t7 which contains the list of all
--  ImageNet training and validation images and their classes. This script also
--  works for other datasets arragned with the same layout.
--

local sys = require 'sys'
local ffi = require 'ffi'
local tds = require 'tds'

local M = {}

local function findImages(dataInfo)
   local imagePath = torch.CharTensor()
   local imageClass = torch.LongTensor()

   local maxLength = -1
   local imagePaths = tds.Vec()
   local imageClasses = tds.Vec()

   for d = 1, #dataInfo do
      f = io.open(paths.concat("./datasets", dataInfo[d][2]), 'r')
      local dir = dataInfo[d][1]

      -- Generate a list of all the images and their class
      while true do
         local line = f:read('*line')
         if not line then break end

         local spaceInd = string.find(line, " ")
         local bpath = line:sub(1,spaceInd-1)
         if bpath:sub(1,1) == '/' then
            bpath = bpath:sub(2,bpath:len())
         end
         local path = paths.concat(dir, bpath)

         local classId = tonumber(line:sub(spaceInd+1, line:len())) + 1
         assert(classId, 'class not found: ' .. path)

         imagePaths:insert(path)
         imageClasses:insert(classId)

         maxLength = math.max(maxLength, #path + 1)
      end

      f:close()
   end

   -- Convert the generated list to a tensor for faster loading
   local nImages = #imagePaths
   local imagePath = torch.CharTensor(nImages, maxLength):zero()
   for i, path in ipairs(imagePaths) do
      ffi.copy(imagePath[i]:data(), path)
   end

   --local imageClass = torch.LongTensor(imageClasses)
   return imagePath, imageClasses
end

function M.exec(opt, cacheFile)
   -- find the image path names
   local imagePath = torch.CharTensor()  -- path to each image in dataset
   local imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)

   local trainDir = paths.concat(opt.data, 'data_large')
   local valDir = paths.concat(opt.data, 'val_large')
   local testDir = paths.concat(opt.data, 'test_large')
   assert(paths.dirp(trainDir), 'train directory not found: ' .. trainDir)
   assert(paths.dirp(valDir), 'val directory not found: ' .. valDir)
   assert(paths.dirp(testDir), 'test directory not found: ' .. testDir)

   print(" | finding all validation images")
   local valImagePath, valImageClass = findImages({{trainDir, "extra_val.txt"},
                                                   {valDir, "places365_val.txt"}, {testDir, 'places365_test.txt'}})

   print(" | finding all training images")
   local trainImagePath, trainImageClass = findImages({{trainDir, "places365_train_challenge_minus_extraval.txt"}})

   local info = {
      basedir = opt.data,
      train = {
         imagePath = trainImagePath,
         imageClass = trainImageClass,
      },
      val = {
         imagePath = valImagePath,
         imageClass = valImageClass,
      },
   }

   print(" | saving list of images to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
