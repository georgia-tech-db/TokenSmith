export async function choosePracticeTopic(page, title = 'B+ trees') {
  const practice = page.locator('.practice-frame')
  await practice.getByLabel('Your topic', { exact: true }).fill(title)
  await practice.getByRole('button', { name: 'Start practice', exact: true }).click()
}

export const idlePractice = page => page.waitForFunction(() => document.querySelector('.practice-frame')?.getAttribute('aria-busy') === 'false')

export async function reviewPracticeQuestion(page, number) {
  await idlePractice(page)
  const current = await page.evaluate(() => { const { practice } = window.practiceTest.state(); return practice.sessions.find(s => s.id === practice.activeSessionId)?.currentIndex })
  if (current === number - 1) return
  const navigation = page.locator('.practice-frame').getByRole('navigation', { name: 'Questions in this practice' })
  await navigation.getByRole('button', { name: new RegExp(`^Question ${number}:`) }).click()
}

export async function answerThenReview(page, answer) {
  const index = await page.evaluate(() => {
    const { practice } = window.practiceTest.state()
    return practice.sessions.find(session => session.id === practice.activeSessionId).currentIndex
  })
  await page.locator('.practice-composer').getByRole('button', { name: answer, exact: true }).click()
  await idlePractice(page)
  await reviewPracticeQuestion(page, index + 1)
}

export async function advancePractice(page, count) {
  const total = await page.evaluate(() => {
    const { practice } = window.practiceTest.state()
    return practice.sessions.find(session => session.id === practice.activeSessionId).questions.length
  })
  if (total < count) {
    await reviewPracticeQuestion(page, total)
    await page.locator('.practice-composer').getByRole('button', { name: 'True', exact: true }).click()
    await page.waitForFunction(count => {
      const { practice } = window.practiceTest.state()
      return practice.sessions.find(session => session.id === practice.activeSessionId).questions.length === count
    }, count)
  }
  await reviewPracticeQuestion(page, count)
}
