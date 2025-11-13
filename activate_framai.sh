#!/bin/bash
# Activation script for FRAMAI
# Usage: source activate_framai.sh

# Activate the virtual environment
source ~/Environments/framai/bin/activate

# Set up convenient alias
alias fram="python3 $PWD/fram_cli.py"

# Display info
echo "============================================================"
echo "                FRAMAI Environment Activated                "
echo "============================================================"
echo ""
echo "Virtual Environment: ~/Environments/framai"
echo "Python Version: $(python --version)"
echo ""
echo "Quick Commands:"
echo "  fram --help                    # Show help"
echo "  fram images ./recordings/      # Process images"
echo "  fram transcribe ./recordings/  # Transcribe audio"
echo "  fram refine fram.json          # Refine with GPT-4"
echo "  fram postprocess ./recordings/ # Post-process audio"
echo ""
echo "Or use the full command:"
echo "  python3 fram_cli.py [command] [options]"
echo "============================================================"
