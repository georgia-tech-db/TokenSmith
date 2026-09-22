import assert from 'node:assert/strict'
import { runBuzzdbMultiturnBenchmark } from '../benchmarks/test_buzzdb_multiturn.mjs'

const result = await runBuzzdbMultiturnBenchmark()
assert.equal(result.passed, result.total, JSON.stringify(result.cases.filter((item) => !item.passed)))
console.log(`Conversation pipeline contracts passed: ${result.passed}/${result.total}. Resolver/search are mocked; this does not measure model accuracy.`)
