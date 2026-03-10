```bash
conda create -n finagentv2
```


```bash
conda activate finagentv2
```


```bash
git clone https://github.com/lakshya-aga/data-mcp
cd data-mcp
pip install -r requirements.txt
pip install -e .
```

```bash
cd ..
```

```bash
git clone https://github.com/lakshya-aga/fin-kit
cd fin-kit
pip install -r requirements.txt
pip install -e .
```


```bash
export OPENAI_API_KEY=sk-proj-...
```

```bash
python run_finagent.py
```
