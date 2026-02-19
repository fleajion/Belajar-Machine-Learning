# 💻 VS Code Setup & Usage Guide

Panduan lengkap menggunakan project ML Pose Classification di VS Code.

## 🚀 Getting Started

### 1. Open Project di VS Code

**Method 1: Command Line**
```bash
cd pose-ml-project
code .
```

**Method 2: VS Code**
- File → Open Folder → Pilih folder project
- Atau: File → Open Workspace → Pilih `pose-ml-project.code-workspace`

### 2. Install Recommended Extensions

VS Code akan prompt untuk install extensions. Klik **Install All**.

Extensions yang direkomendasikan:
- **Python** - IntelliSense, debugging
- **Pylance** - Fast type checking
- **Jupyter** - Notebook support
- **GitLens** - Git integration

Manual install:
1. `Ctrl+Shift+X` (View Extensions)
2. Search: "Python"
3. Install: Microsoft Python extension

## ⚙️ Initial Setup

### Setup Virtual Environment

**Method 1: VS Code Task**
1. `Ctrl+Shift+P` → "Tasks: Run Task"
2. Pilih "Setup Virtual Environment"
3. Tunggu hingga selesai

**Method 2: Command Line**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements_ml.txt
```

**Method 3: Quick Start Script**
```bash
python quick_start.py
```

### Select Python Interpreter

1. `Ctrl+Shift+P` → "Python: Select Interpreter"
2. Pilih: `./venv/bin/python` (atau `.\venv\Scripts\python.exe` di Windows)
3. VS Code akan menggunakan venv untuk semua Python files

## 🎯 Running Scripts

### Using Run Button (▶️)

1. Open file Python (e.g., `ml_pose_classifier.py`)
2. Klik **Run** button (▶️) di top-right
3. Atau tekan `F5` untuk debugging

### Using Terminal

1. Open terminal: `` Ctrl+` ``
2. Pastikan venv active (ada `(venv)` di prompt)
3. Run script:
```bash
python ml_pose_classifier.py
```

### Using Tasks

`Ctrl+Shift+P` → "Tasks: Run Task"

Available tasks:
- **Install Dependencies** - Install packages
- **Run Streamlit App** - Launch web interface
- **Collect Training Data** - Collect pose images
- **Train ML Model** - Train classifier
- **Test Webcam Detection** - Test real-time
- **Clean Project** - Remove cache files

## 🐛 Debugging

### Debug Current File

1. Open Python file
2. Set breakpoint: Click left of line number
3. Press `F5` → Select "Python: Current File"
4. Use debug toolbar:
   - Continue `F5`
   - Step Over `F10`
   - Step Into `F11`
   - Step Out `Shift+F11`

### Debug Configurations

Pre-configured debug configs (press `F5`):

1. **Python: ML Pose Classifier**
   - Runs `ml_pose_classifier.py`
   - For training/testing models

2. **Python: Data Augmentation**
   - Runs `data_augmentation.py`
   - For data collection

3. **Python: Streamlit App**
   - Launches Streamlit web interface
   - Auto-opens browser

4. **Python: Current File**
   - Runs currently open file
   - General purpose

### Debug Variables

While debugging:
- **Variables panel**: View all variables
- **Watch panel**: Monitor specific variables
- **Debug Console**: Execute code in context

Example watches:
```python
len(X_train)
model.get_params()
np.unique(y)
```

## 📝 Working with Jupyter Notebook

### Open ML_Tutorial.ipynb

1. Click `ML_Tutorial.ipynb` in Explorer
2. VS Code opens in notebook mode
3. Select kernel: Click "Select Kernel" → Choose venv

### Running Cells

- **Run Cell**: `Ctrl+Enter`
- **Run and Move**: `Shift+Enter`
- **Run All**: Click "Run All" button
- **Clear Outputs**: Click "Clear All Outputs"

### Interactive Mode

1. Right-click Python file
2. "Run Current File in Interactive Window"
3. Runs code cell-by-cell

## 🔍 Code Navigation

### Quick Navigation

| Shortcut | Action |
|----------|--------|
| `Ctrl+P` | Quick file open |
| `Ctrl+Shift+O` | Go to symbol |
| `Ctrl+T` | Go to symbol in workspace |
| `F12` | Go to definition |
| `Alt+F12` | Peek definition |
| `Shift+F12` | Find all references |

### File Structure

`Ctrl+Shift+O` → View all functions/classes in file

Example: Open `ml_pose_classifier.py` and navigate to:
- `PoseFeatureExtractor`
- `extract_landmarks()`
- `train_all_models()`

## ✏️ Editing Features

### IntelliSense

- Auto-complete: `Ctrl+Space`
- Parameter hints: `Ctrl+Shift+Space`
- Quick fix: `Ctrl+.`

### Code Formatting

**Auto-format on save** (already enabled):
```json
"editor.formatOnSave": true
```

**Manual format**: `Shift+Alt+F`

### Refactoring

- **Rename symbol**: `F2`
- **Extract variable**: `Ctrl+Shift+R`
- **Extract method**: `Ctrl+Shift+R`

Example:
```python
# Before
model.fit(X_train, y_train)

# Select code → Ctrl+Shift+R → Extract Method
# After
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
```

## 📊 Terminal Integration

### Multiple Terminals

1. `` Ctrl+Shift+` `` - New terminal
2. Click `+` in terminal panel
3. Split terminal: Click split icon

Use cases:
- Terminal 1: Training model
- Terminal 2: Streamlit app
- Terminal 3: File operations

### Terminal Shortcuts

| Shortcut | Action |
|----------|--------|
| `` Ctrl+` `` | Toggle terminal |
| `Ctrl+Shift+5` | Split terminal |
| `Alt+Up/Down` | Navigate terminals |
| `Ctrl+C` | Kill process |

## 🔧 Workspace Settings

### Custom Settings

`Ctrl+,` → Settings

**Python Settings:**
```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "python.analysis.typeCheckingMode": "basic"
}
```

**Editor Settings:**
```json
{
  "editor.rulers": [88, 120],
  "editor.tabSize": 4,
  "editor.formatOnSave": true
}
```

### Workspace-Specific Settings

File: `.vscode/settings.json`

Already configured:
- Python interpreter path
- File associations
- Excluded files
- Linting rules

## 🎨 Customization

### Theme

`Ctrl+K Ctrl+T` → Select Color Theme

Recommendations:
- Dark+
- Monokai
- One Dark Pro
- Material Theme

### Layout

- **Toggle Sidebar**: `Ctrl+B`
- **Toggle Panel**: `Ctrl+J`
- **Zen Mode**: `Ctrl+K Z`
- **Split Editor**: `Ctrl+\`

## 📦 Extensions Tips

### Useful Additional Extensions

1. **Python Docstring Generator**
   - Auto-generate docstrings
   - `Ctrl+Shift+2` on function

2. **Better Comments**
   - Color-coded comments
   - TODO, FIXME, etc.

3. **Error Lens**
   - Inline error messages
   - See errors immediately

4. **Code Spell Checker**
   - Spell check in code/comments
   - Catch typos

Install: `Ctrl+Shift+X` → Search → Install

## 🚨 Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'cv2'`

**Solution**:
1. Check interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
2. Select venv interpreter
3. Restart VS Code
4. Or: `` Ctrl+` `` → `pip install opencv-python`

### Linting Errors

**Problem**: Too many linting warnings

**Solution**:
```json
// settings.json
{
  "python.linting.pylintArgs": [
    "--disable=C0111",  // Missing docstring
    "--max-line-length=120"
  ]
}
```

### IntelliSense Not Working

**Solutions**:
1. Reload window: `Ctrl+Shift+P` → "Reload Window"
2. Rebuild IntelliSense: `Ctrl+Shift+P` → "Python: Clear Cache"
3. Check Python path in settings

### Terminal Not Activating venv

**Windows**:
```powershell
# Enable scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Linux/Mac**:
```bash
# Check shell
echo $SHELL
# Might need to use: source venv/bin/activate
```

## 🎓 Workflow Examples

### Complete Training Workflow

1. Open `ml_pose_classifier.py`
2. Set breakpoint at line with `train_all_models()`
3. Press `F5`
4. Choose option `3` (Train models)
5. Step through code
6. Inspect variables
7. View results in `results/` folder

### Data Collection Workflow

1. Open terminal: `` Ctrl+` ``
2. Run: `python data_augmentation.py`
3. Split terminal: Click split icon
4. In new terminal: `python ml_pose_classifier.py`
5. Train immediately after collection

### Web Development Workflow

1. Terminal 1: `streamlit run app_streamlit.py`
2. Terminal 2: Watch for file changes
3. Edit `app_streamlit.py`
4. Streamlit auto-reloads
5. Test changes in browser

## 📚 Resources

### VS Code Documentation
- [Python in VS Code](https://code.visualstudio.com/docs/python/python-tutorial)
- [Debugging](https://code.visualstudio.com/docs/editor/debugging)
- [Jupyter Notebooks](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)

### Keyboard Shortcuts
- `Ctrl+K Ctrl+S` - Keyboard shortcuts reference
- [PDF Cheat Sheet](https://code.visualstudio.com/shortcuts/keyboard-shortcuts-windows.pdf)

### Project-Specific
- `README_ML.md` - Project documentation
- `INSTALLATION.md` - Setup guide
- `ML_Tutorial.ipynb` - Interactive tutorial

---

## 💡 Pro Tips

1. **Multi-cursor editing**: `Alt+Click` or `Ctrl+Alt+Up/Down`
2. **Command palette**: `Ctrl+Shift+P` is your friend
3. **Quick open**: `Ctrl+P` for instant file access
4. **Zen mode**: `Ctrl+K Z` for distraction-free coding
5. **Integrated git**: `Ctrl+Shift+G` for source control
6. **Search in files**: `Ctrl+Shift+F` to search entire project
7. **Rename symbol**: `F2` to rename across all files
8. **Peek definition**: `Alt+F12` to view without switching files

---

**Happy Coding in VS Code! 🚀**
