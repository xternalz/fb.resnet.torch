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
local lightningmdb = require 'lightningmdb'
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

   -- db and keys
   self.env = lightningmdb.env_create()
   local LMDB_MAP_SIZE = 1099511627776 -- 1 TB
   self.env:set_mapsize(LMDB_MAP_SIZE)
   local flags = lightningmdb.MDB_RDONLY + lightningmdb.MDB_NOTLS
   local db, err = self.env:open(self.dir, flags, 0664)
   if not db then
     -- unable to open Database => this might be due to a permission error on
     -- the lock file so we will try again with MDB_NOLOCK. MDB_NOLOCK is safe
     -- in this process as we are opening the database in read-only mode.
     -- However if another process is writing into the database we might have a
     -- concurrency issue - note that this shouldn't happen in DIGITS since the
     -- database is written only once during dataset creation
     print('opening LMDB database failed with error: "' .. err .. '". Trying with MDB_NOLOCK')
     flags = bit.bor(flags, lightningmdb.MDB_NOLOCK)
     -- we need to close/re-open the LMDB environment
     self.env:close()
     self.env = lightningmdb.env_create()
     self.env:set_mapsize(LMDB_MAP_SIZE)
     db, err = self.env:open(self.dir, flags, 0664)
     if not db then
         error('opening LMDB database failed with error: ' .. err)
     end
   end
   self.total = self.env:stat().ms_entries
   self.txn = self.env:txn_begin(nil, lightningmdb.MDB_RDONLY)
   self.d = self.txn:dbi_open(nil,0)
   self.keys = self.lmdbInfo[self.split].Keys
   assert(#self.keys == self.total, 'failed to initialize DB - #keys=' .. #self.keys .. ' #records='.. self.total)
end

function ImagenetLMDBDataset:get(idx)
   local key = self.keys[idx]
   local v = self.txn:get(self.d, key, lightningmdb.MDB_FIRST)
   assert(key~=nil, "lmdb read nil key at idx="..idx)
   assert(v~=nil, "lmdb read nil value at idx="..idx.." key="..key)

   -- label
   local _,labelInd = string.find(key, '~.~')
   local label = tonumber(key:sub(labelInd+1,key:len())) + 1

   return {
      input = image.decompress(x, 3, 'float'),
      target = label,
   }
end

function ImagenetLMDBDataset:close()
   self.total = 0
   self.env:dbi_close(self.d)
   self.txn:abort()
   self.env:close()
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
