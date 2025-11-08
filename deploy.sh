#!/bin/bash
#
# Automated deployment script
# Run this from your LOCAL machine to transfer all files to VM
#

set -e

ZONE="us-central1-a"
INSTANCE="wikipedia-embeddings"

echo "=========================================="
echo "Deploying Wikipedia Embeddings System"
echo "=========================================="
echo ""

# Check if instance exists
echo "Checking if instance '$INSTANCE' exists..."
if ! gcloud compute instances describe $INSTANCE --zone=$ZONE &>/dev/null; then
    echo "ERROR: Instance '$INSTANCE' not found in zone $ZONE"
    echo ""
    echo "Create it first with:"
    echo "  gcloud compute instances create wikipedia-embeddings \\"
    echo "    --zone=us-central1-a \\"
    echo "    --machine-type=n1-standard-8 \\"
    echo "    --accelerator=type=nvidia-tesla-t4,count=1 \\"
    echo "    --boot-disk-size=100GB \\"
    echo "    --create-disk=name=wikipedia-data,size=200GB,type=pd-balanced \\"
    echo "    --image=pytorch-2-7-cu128-ubuntu-2204-nvidia-570-v20251107 \\"
    echo "    --image-project=deeplearning-platform-release \\"
    echo "    --maintenance-policy=TERMINATE \\"
    echo "    --preemptible \\"
    echo "    --metadata=\"install-nvidia-driver=True\""
    exit 1
fi

echo "✓ Instance found"
echo ""

# Transfer files
echo "Transferring files to VM..."
echo ""

files=(
    "generate_wikipedia_embeddings.py"
    "setup.sh"
    "monitor.py"
    "preflight.py"
    "requirements.txt"
    "README.md"
    "DEPLOYMENT_GUIDE.md"
    "EXECUTIVE_SUMMARY.md"
)

for file in "${files[@]}"; do
    if [ ! -f "/tmp/$file" ]; then
        echo "WARNING: /tmp/$file not found, skipping"
        continue
    fi
    
    echo "  Uploading $file..."
    gcloud compute scp "/tmp/$file" "$INSTANCE:/tmp/" --zone=$ZONE --quiet
done

echo ""
echo "✓ All files transferred"
echo ""

# SSH in and organize files
echo "Organizing files on VM..."
gcloud compute ssh $INSTANCE --zone=$ZONE --command="
    set -e
    echo 'Moving files to home directory...'
    cd /tmp
    sudo mv setup.sh /home/\$(whoami)/ 2>/dev/null || true
    sudo mv *.py /home/\$(whoami)/ 2>/dev/null || true
    sudo mv *.md /home/\$(whoami)/ 2>/dev/null || true
    sudo mv *.txt /home/\$(whoami)/ 2>/dev/null || true
    cd /home/\$(whoami)
    chmod +x setup.sh
    ls -lh
    echo ''
    echo '✓ Files organized'
"

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. SSH into the VM:"
echo "   gcloud compute ssh $INSTANCE --zone=$ZONE"
echo ""
echo "2. Run setup:"
echo "   cd ~"
echo "   ./setup.sh"
echo ""
echo "3. Run pre-flight checks:"
echo "   python3 preflight.py"
echo ""
echo "4. Start the pipeline:"
echo "   python3 generate_wikipedia_embeddings.py"
echo ""
echo "5. In another terminal, monitor progress:"
echo "   gcloud compute ssh $INSTANCE --zone=$ZONE"
echo "   python3 monitor.py"
echo ""
echo "See DEPLOYMENT_GUIDE.md for detailed instructions"
echo ""