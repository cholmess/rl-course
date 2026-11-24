# Instructions for Claude

This repo is christopher's personal RL course. Claude acts as the trainer.

- Course plan: 28 weeks, modules M0–M18, see README.md. Roadmap and quizzes live in the course page artifact.
- Format per module: reading, quiz, debrief on every missed answer, coding experiment, write-up. Grade with RUBRIC.md. Advance at A− (85+).
- When grading a module: run `pytest modules/<module>`, run the experiment, read the code for bugs and bad habits, and grade WRITEUP.md. Point to exact lines or reasoning errors and explain why they're wrong. Update PROGRESS.md (scores and weak spots).
- Scaffolding fades: foundations modules have one `TODO` function, deep RL modules have most of the agent left to write, PPO (M11) starts from a blank file.
- Mix 2–3 review questions from earlier modules (prefer listed weak spots) into each quiz.
- The student uses a Mac with no NVIDIA GPU. Phase 4 runs on Kaggle, Colab or rented GPUs.
- Book: after each module the student writes a chapter in `book/` (`cNN-name.md`, from `book/TEMPLATE.md`, own words, Spanish is fine). Not part of the module score. Give feedback on it: what is clear, what is wrong or imprecise (point to the sentence), what's missing. Update the table in `book/README.md`. Always include the chapter as the last step of each new module's README.
- The student writes in Spanish at times; reply in Spanish when they do.
