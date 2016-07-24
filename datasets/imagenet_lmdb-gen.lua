--
--  Copyright (c) 2016, Jason Kuen.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script to compute list of ImageNet filenames and classes
--
--  This generates a file gen/imagenet_lmdb.t7 which contains information about
--  ImageNet LMDB records. This script also works for other datasets
--  arragned with the same layout.
--

local M = {}

local function getKeys(path, split)
   local Keys
   local tds = require 'tds'
   Keys = tds.Vec()
   local lmdb = require 'lmdb'
   local db = lmdb.env{Path = path, Name = split}
   db:open()
   local txn = db:txn(true)
   local cursor = txn:cursor()
   local numEntries = db:stat().entries
   for i = 1, numEntries do
      local key = select(1, cursor:get())
      Keys[i] = key
      if i < numEntries then
         cursor:next()
      end
   end
   cursor:close()
   db:close()
   return Keys
end

function M.exec(opt, cacheFile)

   -- Caffe LMDB datum
   local datumFile = io.open(paths.concat(opt.data, 'datum.proto'))
   local datumText = datumFile:read("*a")
	datumFile:close()

   -- Image channels, width, height
   local img_dim_file = torch.DiskFile(paths.concat(opt.data, 'img_dim.txt'), 'r')
   local channels = img_dim_file:readInt()
   local width = img_dim_file:readInt()
   local height = img_dim_file:readInt()
   img_dim_file:close()

   local info = {
      basedir = opt.data,
      datumText = datumText,
      channels = channels,
      width = width,
      height = height,
      train = {
         Keys = getKeys(paths.concat(opt.data, 'train'), 'train')
      },
      val = {
         Keys = getKeys(paths.concat(opt.data, 'val'), 'val')
      },
   }

   print(" | saving info to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
