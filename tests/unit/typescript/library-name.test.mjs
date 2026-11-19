import assert from 'node:assert/strict'
import test from 'node:test'
import { requireTranspiledTs } from './ts-module-loader.mjs'
const {uniqueLibraryName,libraryNameKey}=requireTranspiledTs('src/shared/library-name.ts')
test('folder names receive the first unused suffix across case and whitespace variants',()=>{
 assert.equal(uniqueLibraryName('Biology',[' BIOLOGY ','biology (2)']),'Biology (3)')
 assert.equal(uniqueLibraryName('Physics',['Biology']),'Physics')
 assert.equal(libraryNameKey('  Study   Notes '),libraryNameKey('study notes'))
 assert.equal(libraryNameKey('Ｓｔｕｄｙ'),libraryNameKey('Study'))
})
