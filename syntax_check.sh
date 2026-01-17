#!/bin/bash

echo "====================================="
echo "Syntax Check for All Modified Files"
echo "====================================="

FILES=(
    "src/services/reward_models.py"
    "src/methodes/twinflow/twinflow.py"
    "src/steerers/qwenimage/sft_fsdp.py"
)

FAIL=0

for file in "${FILES[@]}"; do
    echo ""
    echo "Checking: $file"
    python -m py_compile "$file" 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ $file - OK"
    else
        echo "✗ $file - SYNTAX ERROR"
        FAIL=1
    fi
done

echo ""
echo "====================================="
if [ $FAIL -eq 0 ]; then
    echo "✓ All files passed syntax check"
else
    echo "✗ Some files have syntax errors"
fi
echo "====================================="

exit $FAIL