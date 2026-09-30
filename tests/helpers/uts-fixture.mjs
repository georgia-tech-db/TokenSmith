// Synthetic "University of TokenSmith" (UTS) fixture: a test example, not textbook content.
export const utsRows = {
  Department: [[101, 'Computer Sci.', 'Building A'], [102, 'Engineering', 'Building B'], [103, 'Business', 'Downtown']],
  Course: [['CSE101', 'Intro to CS 1', 3, 101], ['ENG202', 'Engineering Math', 4, 102], ['BUS301', 'Marketing', 3, 103]],
  Instructor: [['I1', 'John Smith', 101], ['I2', 'Jane Doe', 102], ['I3', 'Bob Johnson', 103]]
}

const fmt = (row) => `(${JSON.stringify(row).slice(1, -1)})`
// The student's saving message, as typed. It is stored, never answered.
export const utsSaveMessage = [
  'Use this consistent fixture for development. Call the example University of TokenSmith:',
  'Department(dept_id PRIMARY KEY, name, location)',
  ...utsRows.Department.map((row) => `  ${fmt(row)}`),
  '',
  'Course(course_id PRIMARY KEY, title, credits, dept_id)',
  ...utsRows.Course.map((row) => `  ${fmt(row)}`),
  '',
  'Instructor(instructor_id PRIMARY KEY, name, dept_id)',
  ...utsRows.Instructor.map((row) => `  ${fmt(row)}`),
  '',
  'Course.dept_id      REFERENCES Department.dept_id',
  'Instructor.dept_id  REFERENCES Department.dept_id'
].join('\n')

export const utsRecallQuestion = 'Without restating the schema, list all nine rows in our running example. ' +
  "Identify each relation's primary key and foreign keys, and explain how they connect."
export const utsUnrelatedQuestion = 'What is two-phase locking?'
