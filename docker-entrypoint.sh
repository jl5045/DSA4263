#!/bin/bash
set -e

echo "🚀 Starting Docker entrypoint script..."

# Check if data directory already exists and has content
if [ -d "data" ] && [ "$(ls -A data)" ]; then
    echo "✅ Data directory already exists, skipping extraction"
# Check if data-docker directory is already mounted (volume mount for production)
elif [ -d "data-docker" ] && [ "$(ls -A data-docker)" ]; then
    echo "✅ data-docker directory is mounted as volume"
    echo "📁 Creating symlink: data -> data-docker"
    ln -sf data-docker data
    echo "✅ Symlink created, using volume-mounted data"
# Otherwise, try to extract from zip if it exists
elif [ -f "data-docker.zip" ]; then
    echo "📦 Extracting data-docker.zip to ./data/ ..."
    unzip -q data-docker.zip
    # Rename extracted folder to match expected path
    if [ -d "data-docker" ]; then
        mv data-docker data
        echo "✅ Data extraction complete! (data-docker -> data)"
    else
        echo "⚠️  Warning: Expected data-docker folder not found in zip"
    fi
else
    echo "⚠️  Warning: No data source found (neither volume mount nor data-docker.zip)"
    echo "    The application may not work correctly without data."
fi

echo "🎯 Starting Streamlit application..."
# Execute the main command (passed as arguments to this script)
exec "$@"
