# DEPLOY.md

## Option A — Streamlit Cloud
1. Fork/upload repo to GitHub.
2. In Streamlit Cloud, create a new app → point to `app.py`.
3. Add `OPENAI_API_KEY` as a secret.
4. Deploy.

## Option B — AWS EC2 (Ubuntu)
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
streamlit run app.py --server.port 80 --server.address 0.0.0.0
```
Open security group for HTTP (80). Access via EC2 public DNS.

## Option C — Docker on EC2
- See README's Dockerfile.
- Expose 8501 on the instance SG and run the container.