mkdir -p ~/.streamlit/

echo "\
[server]
port = 8501
enableCORS = false
headless = true
" > ~/.streamlit/config.toml
