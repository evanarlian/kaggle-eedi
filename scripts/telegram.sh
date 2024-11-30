#!/usr/bin/env bash

curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
  -d chat_id=$TELEGRAM_CHAT_ID \
  -d text="$1"
