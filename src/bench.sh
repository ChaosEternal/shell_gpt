
for m in granite-embedding:278m embeddinggemma:300m snowflake-arctic-embed2:latest nomic-embed-text:latest; do
    echo ======= $m
    echo update agent_messages set embedding=null|sqlite3 /tmp/session.db
    time python3 -m sadk.session /tmp/session.db "get current time" "use command date" "time" $m
done
