--
--  Copyright (c) 2016, Jason Kuen.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ImageNet LMDB dataset loader
--

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local lmdb = require "lmdb"
local pb = require 'pb'
local ffi = require 'ffi'
local tds = require 'tds'

local M = {}
local ImagenetLMDBDataset = torch.class('resnet.ImagenetLMDBDataset', M)

function ImagenetLMDBDataset:__init(lmdbInfo, opt, split)
   self.opt = opt
   self.lmdbInfo = lmdbInfo
   self.split = split
   self.dir = paths.concat(opt.data, self.split)
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)

   -- caffe datum
   self.datum = pb.load_proto(self.lmdbInfo.datumText, paths.concat(opt.data, 'datum'))

   -- keys and db
   self.keys = self.lmdbInfo[self.split].Keys
   self.db = lmdb.env{Path = self.dir, Name = split}
   self.db:open()
   self.total = self.db:stat().entries
   assert(#self.keys == self.total, 'failed to initialize DB - #keys=' .. #self.keys .. ' #records='.. self.total)
   self.reader = self.db:txn(true)

   -- Image dimensions
   self.ImageChannels = self.lmdbInfo.channels
   self.ImageSizeX = self.lmdbInfo.width
   self.ImageSizeY = self.lmdbInfo.height
end

function ImagenetLMDBDataset:get(idx)
   local label

   local key = self.keys[idx]
   local v = self.reader:get(key)
   assert(key~=nil, "lmdb read nil key at idx="..idx)
   assert(v~=nil, "lmdb read nil value at idx="..idx.." key="..key)

   local total = self.ImageChannels*self.ImageSizeY*self.ImageSizeX
   -- Tensor allocations inside loop consumes little more execution time. So allocated "x" outside with double size of an image and inside loop if any encoded image is encountered with bytes size more than Tensor size, then the Tensor is resized appropriately.
   local x = torch.ByteTensor(total*2):contiguous() -- sometimes length of JPEG files are more than total size. So, "x" is allocated with more size to ensure that data is not truncated while copying.
   local x_size = total * 2 -- This variable is just to avoid the calls to tensor's size() i.e., x:size(1)
   local temp_ptr = torch.data(x) -- raw C pointer using torchffi

   local msg = self.datum.Datum():Parse(v)

   -- label
   label = msg.label + 1

   if #msg.data > x_size then
     x:resize(#msg.data+1) -- 1 extra byte is required to copy zero terminator i.e., '\0', by ffi.copy()
     x_size = #msg.data
   end

   ffi.copy(temp_ptr, msg.data)

   local y=nil
   if msg.encoded==true then
     y = image.decompress(x,self.ImageChannels,'byte'):float()
   else
     x = x:narrow(1,1,total):view(self.ImageChannels,self.ImageSizeY,self.ImageSizeX):float() -- using narrow() returning the reference to x tensor with the size exactly equal to total image byte size, so that view() works fine without issues
     if self.ImageChannels == 3 then
         -- unencoded color images are stored in BGR order => we need to swap blue and red channels (BGR->RGB)
         y = torch.FloatTensor(self.ImageChannels,self.ImageSizeY,self.ImageSizeX)
         y[1] = x[3]
         y[2] = x[2]
         y[3] = x[1]
     else
         y = x
     end
   end

   -- Normalize to range 0,1
   if torch.max(y) > 1 then
      y:div(255)
   end

   return {
      input = y,
      target = label,
   }
end

function ImagenetLMDBDataset:close()
   self.reader:abort()
   self.db:close()
end

function ImagenetLMDBDataset:size()
   return self.total
end

-- Computed from random subset of ImageNet training images
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}

function ImagenetLMDBDataset:preprocess()
   if self.split == 'train' then
      return t.Compose{
         t.RandomSizedCrop(224),
         t.ColorJitter({
            brightness = 0.4,
            contrast = 0.4,
            saturation = 0.4,
         }),
         t.Lighting(0.1, pca.eigval, pca.eigvec),
         t.ColorNormalize(meanstd),
         t.HorizontalFlip(0.5),
      }
   elseif self.split == 'val' then
      local Crop = self.opt.tenCrop and t.TenCrop or t.CenterCrop
      return t.Compose{
         t.Scale(256),
         t.ColorNormalize(meanstd),
         Crop(224),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.ImagenetLMDBDataset
