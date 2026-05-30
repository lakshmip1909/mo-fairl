#!/usr/bin/env bash
# =============================================================================
# GITHUB_SETUP.sh
#
# Complete step-by-step guide to push mo-fairl to GitHub from terminal.
# Run each section manually (don't run the whole script blindly).
# =============================================================================

# =============================================================================
# PART 1: ONE-TIME GITHUB SETUP (skip if already done)
# =============================================================================

# 1a. Install GitHub CLI (gh) — easiest way to authenticate
#     On Ubuntu/Debian:
sudo apt install gh -y
#     On Mac:
# brew install gh

# 1b. Log into GitHub via CLI
gh auth login
# Select: GitHub.com → HTTPS → Login with browser
# Follow the browser prompt and paste the code shown

# 1c. Configure git identity (if not done already)
git config --global user.name  "Your Name"
git config --global user.email "your@email.com"

# =============================================================================
# PART 2: CREATE THE GITHUB REPO
# =============================================================================

# 2a. Create repo on GitHub (public, with description, no auto-init)
gh repo create mo-fairl \
    --public \
    --description "Multi-Objective Failure-Aware Inverse Reward Learning for LLM alignment" \
    --confirm

# This prints the repo URL, e.g.:
#   https://github.com/YOUR_USERNAME/mo-fairl

# =============================================================================
# PART 3: INITIALISE LOCAL GIT REPO
# =============================================================================

# 3a. Go into the project folder
cd mo-fairl       # wherever you cloned/built the project

# 3b. Initialise git
git init

# 3c. Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/
.env
venv/
env/

# Data (large files — don't commit raw data)
data/*.jsonl

# Outputs (generated, don't commit)
outputs/checkpoints/
outputs/metrics/
outputs/plots/
outputs/failure_examples/

# Logs
logs/
*.log

# Editor
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
EOF

# 3d. Stage everything
git add .

# 3e. First commit
git commit -m "Initial commit: MO-FAIRL project structure and full pipeline"

# =============================================================================
# PART 4: PUSH TO GITHUB
# =============================================================================

# 4a. Add the remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/mo-fairl.git

# 4b. Set main branch and push
git branch -M main
git push -u origin main

# You should see something like:
#   Enumerating objects: 25, done.
#   To https://github.com/YOUR_USERNAME/mo-fairl.git
#    * [new branch]      main -> main

# =============================================================================
# PART 5: VERIFY
# =============================================================================

# 5a. Open in browser to confirm
gh repo view --web

# =============================================================================
# PART 6: ON YOUR HPC — CLONE AND SET UP
# =============================================================================

# On your PBS cluster, after SSHing in:

# 6a. Clone the repo
git clone https://github.com/YOUR_USERNAME/mo-fairl.git
cd mo-fairl

# 6b. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 6c. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 6d. Create log directory (PBS jobs write here)
mkdir -p logs data outputs/checkpoints outputs/metrics outputs/plots outputs/failure_examples

# =============================================================================
# PART 7: RUNNING THE PIPELINE ON PBS
# =============================================================================

# Submit jobs in order (each depends on the previous)

# Step 1: Generate data
qsub jobs/generate_pairs.pbs

# Check job status
qstat -u $USER

# Step 2: Train (after generation job finishes)
qsub jobs/train_mo_fairl.pbs

# Step 3: Evaluate
qsub jobs/evaluate_mo_fairl.pbs

# Step 4: Failure analysis
qsub jobs/failure_analysis.pbs

# OR: chain jobs with dependency (job 2 starts after job 1 finishes)
# JOB1=$(qsub jobs/generate_pairs.pbs)
# JOB2=$(qsub -W depend=afterok:$JOB1 jobs/train_mo_fairl.pbs)
# JOB3=$(qsub -W depend=afterok:$JOB2 jobs/evaluate_mo_fairl.pbs)
# JOB4=$(qsub -W depend=afterok:$JOB3 jobs/failure_analysis.pbs)

# =============================================================================
# PART 8: LOCAL RUN (no PBS, no GPU — for testing)
# =============================================================================

# First edit configs/default.yaml:
#   training:
#     device: cpu
#     num_epochs: 3
#     batch_size: 16

# Then:
python scripts/generate_toxicity.py
python scripts/generate_math.py
python scripts/generate_code.py
python scripts/combine_data.py
python main.py --config configs/default.yaml --mode train
python main.py --config configs/default.yaml --mode evaluate
python main.py --config configs/default.yaml --mode analyze

# =============================================================================
# PART 9: PUSHING UPDATES BACK TO GITHUB
# =============================================================================

# After making changes:
git add .
git commit -m "Add: meaningful description of change"
git push

# Pull latest changes on HPC:
git pull

# =============================================================================
# DONE.
# Your repo is at: https://github.com/YOUR_USERNAME/mo-fairl
# =============================================================================
