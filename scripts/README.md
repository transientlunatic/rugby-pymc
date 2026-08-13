# Visualization Scripts

This directory contains scripts for generating diagrams and visualizations for the rugby ranking model.

## Plate Notation Diagrams

### Requirements

```bash
pip install daft-pgm matplotlib
```

**Important**: Install `daft-pgm` (not just `daft`). The `daft` package on PyPI is different.

### Scripts

#### `create_plate_diagram.py` - Full Model Diagram

Generates a comprehensive plate notation diagram showing all components of the hierarchical model.

**Usage:**
```bash
python scripts/create_plate_diagram.py
```

**Outputs:**
- `rugby_model_plate_diagram.pdf` - High quality for papers/printing
- `rugby_model_plate_diagram.png` - For web/blog (300 DPI)
- `rugby_model_plate_diagram.svg` - Scalable vector graphics

**Features:**
- Shows all hyperparameters
- Separate try-scoring and kicking effects
- Score-type specific parameters
- Loading factors
- Player and team-season plates
- Observation level with linear predictor

#### `create_simple_plate_diagram.py` - Simplified Version

Generates a cleaner, more digestible version for blog posts and presentations.

**Usage:**
```bash
python scripts/create_simple_plate_diagram.py
```

**Outputs:**
- `rugby_model_simple.pdf`
- `rugby_model_simple.png`
- `rugby_model_simple.svg`

**Features:**
- Abstracted hyperparameters
- Clear player/team separation
- Minimal clutter
- Focus on key hierarchical structure

### Customization

Both scripts are well-commented and easy to customize:

- **Layout**: Adjust `shape` and `grid_unit` in `daft.PGM()` constructor
- **Node positions**: Modify x,y coordinates in `add_node()` calls
- **Styling**: Change `plot_params` for colors, line widths, etc.
- **Plates**: Adjust plate positions and labels in `add_plate()` calls
- **Text**: Modify annotations and captions

### Example Customizations

**Change colors:**
```python
# Make player nodes blue
pgm.add_node("beta_try", r"$\beta^{\mathrm{try}}_i$", x, y,
             plot_params={"ec": "blue", "fc": "lightblue"})
```

**Adjust spacing:**
```python
# Make diagram wider
pgm = daft.PGM(shape=[10, 8], grid_unit=2.5)  # wider grid
```

**Add custom annotations:**
```python
plt.text(x, y, "Your text here",
         fontsize=10, ha='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
```

## Tips for Blog Use

1. **PNG for inline images**: Use the PNG output at 300 DPI for sharp rendering
2. **SVG for interactive**: Use SVG if your blog supports it for perfect scaling
3. **PDF for downloads**: Include PDF as downloadable supplement for readers
4. **Simplified version first**: Show the simple diagram in the main text, link to detailed version
5. **Dark mode**: Consider generating both light and dark background versions:
   ```python
   plt.savefig(filename, facecolor='#1e1e1e')  # dark background
   ```

## Troubleshooting

**LaTeX errors**: If you see LaTeX-related errors, comment out these lines in the script:
```python
# rc("font", family="serif", size=12)
# rc("text", usetex=True)
```

**Daft not installed**:
```bash
pip install daft
```

**Display issues**: If diagrams don't display, ensure you have a working matplotlib backend:
```bash
# For WSL/headless systems
export MPLBACKEND=Agg
python scripts/create_plate_diagram.py
```

## Other Visualization Tools

The scripts also work as templates for:
- **Conference posters**: Increase DPI to 600 for large format printing
- **Presentations**: Export as SVG and import into Keynote/PowerPoint
- **Interactive notebooks**: Embed in Jupyter notebooks with `%matplotlib inline`
