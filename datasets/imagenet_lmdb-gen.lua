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

local function getKeys(path, split)
   local lightningmdb = require 'lightningmdb'
   env = lightningmdb.env_create()
   local LMDB_MAP_SIZE = 1099511627776 -- 1 TB
   env:set_mapsize(LMDB_MAP_SIZE)
   local flags = lightningmdb.MDB_RDONLY + lightningmdb.MDB_NOTLS
   local db, err = env:open(path, flags, 0664)
   if not db then
     -- unable to open Database => this might be due to a permission error on
     -- the lock file so we will try again with MDB_NOLOCK. MDB_NOLOCK is safe
     -- in this process as we are opening the database in read-only mode.
     -- However if another process is writing into the database we might have a
     -- concurrency issue - note that this shouldn't happen in DIGITS since the
     -- database is written only once during dataset creation
     print('opening LMDB database failed with error: "' .. err .. '". Trying with MDB_NOLOCK')
     flags = bit.bor(flags, lighningmdb.MDB_NOLOCK)
     -- we need to close/re-open the LMDB environment
     env:close()
     env = lightningmdb.env_create()
     env:set_mapsize(LMDB_MAP_SIZE)
     db, err = env:open(path, flags, 0664)
     if not db then
         error('opening LMDB database failed with error: ' .. err)
     end
   end
   local total = env:stat().ms_entries
   local txn = env:txn_begin(nil, lightningmdb.MDB_RDONLY)
   local d = txn:dbi_open(nil,0)
   local cursor = txn:cursor_open(d)
   local tds = require 'tds'
   local Keys = tds.Vec()
   local i = 0
   for k,_ in all_keys(cursor,nil,lightningmdb.MDB_NEXT) do
     i=i+1
     Keys[i] = k
   end
   cursor:close()
   env:dbi_close(d)
   txn:abort()
   env:close()
   return Keys
end

function M.exec(opt, cacheFile)

   local info = {
      basedir = opt.data,
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
