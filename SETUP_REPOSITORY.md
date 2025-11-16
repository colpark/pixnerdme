# Setting Up colpark/pixnerdme Repository

This guide will help you create and push the CIFAR-10 fork to your GitHub repository.

## Step 1: Create the GitHub Repository

### Option A: Using GitHub Web Interface

1. Go to https://github.com/new
2. Set the repository name to: `pixnerdme`
3. Set the owner to: `colpark`
4. Choose visibility: Public or Private
5. **Do NOT initialize** with README, .gitignore, or license (we already have these)
6. Click "Create repository"

### Option B: Using GitHub CLI (if installed)

```bash
# Install GitHub CLI if not already installed
# macOS: brew install gh
# Then authenticate
gh auth login

# Create the repository
gh repo create colpark/pixnerdme --public --source=. --remote=origin
```

## Step 2: Update Git Remote

Once the repository is created on GitHub, update your local repository's remote:

```bash
# Remove the old origin (MCG-NJU/PixNerd)
git remote remove origin

# Add your new repository as origin
git remote add origin https://github.com/colpark/pixnerdme.git

# Verify the remote
git remote -v
```

You should see:
```
origin  https://github.com/colpark/pixnerdme.git (fetch)
origin  https://github.com/colpark/pixnerdme.git (push)
```

## Step 3: Push to GitHub

```bash
# Push the main branch
git checkout main
git push -u origin main

# Push the CIFAR-10 feature branch
git checkout feature/cifar10-support
git push -u origin feature/cifar10-support
```

## Step 4: Merge CIFAR-10 Support (Optional)

If you want to merge the CIFAR-10 support into main:

### Option A: Via GitHub Pull Request (Recommended)

1. Go to https://github.com/colpark/pixnerdme
2. Click "Pull requests" → "New pull request"
3. Base: `main`, Compare: `feature/cifar10-support`
4. Review changes and click "Create pull request"
5. Add description and click "Merge pull request"

### Option B: Via Command Line

```bash
# Switch to main branch
git checkout main

# Merge the feature branch
git merge feature/cifar10-support

# Push to GitHub
git push origin main
```

## Step 5: Update README (Optional)

You may want to update the main README.md to mention CIFAR-10 support:

```bash
# Edit README.md to add CIFAR-10 section
# Then commit and push
git add README.md
git commit -m "Update README with CIFAR-10 support"
git push origin main
```

## Repository Structure

After setup, your repository will have:

```
colpark/pixnerdme/
├── main                           # Original PixNerd code
└── feature/cifar10-support        # CIFAR-10 additions
    ├── CIFAR10_README.md          # CIFAR-10 documentation
    ├── configs_c2i/cifar_repa_v1.yaml  # CIFAR-10 config
    ├── src/data/dataset/cifar10.py     # CIFAR-10 dataset
    ├── test_cifar10.py            # Validation script
    └── .gitignore                 # Python ignore patterns
```

## Quick Start After Setup

Once pushed to GitHub, anyone can use it with:

```bash
# Clone the repository
git clone https://github.com/colpark/pixnerdme.git
cd pixnerdme

# Install dependencies
pip install -r requirements.txt

# Train on CIFAR-10
python main.py fit -c configs_c2i/cifar_repa_v1.yaml
```

## Maintaining Attribution

This repository is a fork of the original PixNerd project. Please maintain proper attribution:

1. Keep the original LICENSE file
2. Reference the original paper in publications
3. Link back to MCG-NJU/PixNerd in your README

Original repository: https://github.com/MCG-NJU/PixNerd

## Troubleshooting

### Authentication Issues

If you get authentication errors when pushing:

```bash
# Use SSH instead of HTTPS
git remote set-url origin git@github.com:colpark/pixnerdme.git
```

Or configure GitHub Personal Access Token:
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate new token with `repo` scope
3. Use token as password when pushing

### Permission Denied

Make sure you have write access to the `colpark` organization/account.

If `colpark` is an organization, you need to be a member with appropriate permissions.
