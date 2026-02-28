# 🚀 Git Workflow for PSG_Pipeline

## 🔹 First Setup

GitHub is **remote**, and you work locally (for example in lola).  
So you need to:

1. Clone the repository (remote → local)
2. Create your own branch (locally)
3. Push your branch to GitHub (local → remote)

### 1️⃣ Clone the repository

```bash
git clone https://github.com/AliceNeuro/PSG_Pipeline.git
```

(You can also use SSH if configured.)

### 2️⃣ Create your own branch

```bash
git checkout -b name_of_your_branch
```

### 3️⃣ Push your branch to GitHub

```bash
git push -u origin name_of_your_branch
```

Now your branch exists both locally and remotely.

---

# 🔹 Working on Your Branch

You should always work on **your own branch** (not on `master`).  
Your branch is unprotected, so you can push your changes freely.

### 1️⃣ Make your changes  
Code, edit files, do your magic ✨

### 2️⃣ Add your changes

Add all files:
```bash
git add .
```

Or add a specific file:
```bash
git add path/to/file.py
```

### 3️⃣ Commit your changes

```bash
git commit -m "Short description of what you changed"
```

### 4️⃣ Push your updates

```bash
git push
```

(After the first `-u origin`, you don't need to repeat it.)

---

# 🔹 Stay Updated with Master

We are 3 people working together, so please regularly sync with `master`.

### 1️⃣ Switch to master

```bash
git checkout master
```

### 2️⃣ Pull latest changes from GitHub

```bash
git pull
```

### 3️⃣ Switch back to your branch

```bash
git checkout name_of_your_branch
```

### 4️⃣ Merge master into your branch

```bash
git merge master
```

---

## 🔹 If There Are Merge Conflicts

If Git reports conflicts:

1. Open the conflicted file.
2. Fix the conflict (choose or combine changes).
3. Add the fixed file:

```bash
git add path/to/this_file
```

4. Commit the merge:

```bash
git commit -m "Merge master into my branch"
```

5. Check everything:

```bash
git status
```

6. Push:

```bash
git push
```

---

# 🔹 Create a Pull Request (PR)

When your code is ready and tested, you can propose adding it to `master`.

### Steps:

1. Go to the repository on **github.com**
2. Click **"Compare & pull request"**
3. Submit the pull request

I will review it.

- If changes are needed → I will request changes.
- You fix them and push again.
- We repeat until everything is good.
- Then the PR is merged into `master`.

---

# ✅ Important Rules

- Never work directly on `master`
- Always keep your branch updated with `master`
- Write clear commit messages
- Test your code before opening a Pull Request
