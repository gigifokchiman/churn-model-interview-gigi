# scripts/setup/setup_mac.sh
#!/bin/bash

# Update Homebrew and formulae
echo "Updating Homebrew..."
brew update
brew upgrade

# Install system dependencies
echo "Installing system dependencies..."
brew install libomp
brew install python@3.10
