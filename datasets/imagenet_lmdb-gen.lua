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

local function all_keys(cursor_,key_,op_)
   return coroutine.wrap(
      function()
         local k = key_
         local v
         repeat
            k,v = cursor_:get(k,op_ or MDB.NEXT)
            if k then
              coroutine.yield(k,v)
            end
         until (not k)
      end)
end

local function getKeys(filelists, split)
   local tds = require 'tds'
   local Keys = tds.Vec()
   local k = 1
   for f = 1, #filelists do
      local file = io.open(filelists[f], 'r')
      while true do
        local line = file:read("*line")
        if line == nil then break end
        local separator = string.find(line, " ")
        local key = line:sub(1,separator-1) .. '~.~' .. line:sub(separator+1, line:len())
        Keys[k] = key
        k = k + 1
      end
   end
   return Keys
end

function M.exec(opt, cacheFile)

   local info = {
      basedir = opt.data,
      train = {
         Keys = getKeys({'datasets/places365_train_challenge_minus_extraval.txt'}, 'train')
      },
      val = {
         Keys = getKeys({'datasets/places365_val.txt',
                        'datasets/extra_val.txt'}, 'val')
      },
   }

   print(" | saving info to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
