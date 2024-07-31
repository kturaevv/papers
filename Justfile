default: 
    just -l

# Generate .md files from .ipynb in all folders
update *arg='.':
   find {{arg}} -maxdepth 2 -type f -name "*.ipynb" \
    -exec jupyter nbconvert --output README.md --to markdown {} +
