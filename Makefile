# Makefile for rugby-ranking MCMC training and release management
# Usage:
#   make mcmc              - Train MCMC model locally
#   make mcmc-condor       - Generate HTCondor submit file and submit
#   make upload-release    - Upload checkpoint to GitHub Release
#   make mcmc-and-upload   - Train and upload in one step

# Configuration
DATA_DIR ?= ../Rugby-Data/json
CHECKPOINT_NAME ?= mcmc-$(shell date +%Y-%m)
RELEASE_TAG ?= v$(shell date +%Y.%m)
RELEASE_TITLE ?= "MCMC Checkpoint - $(shell date +%B\ %Y)"

# Training parameters
MCMC_DRAWS ?= 1000
MCMC_CHAINS ?= 4
LAST_SEASONS ?= 5
MODEL_TYPE ?= static

# Paths
CACHE_DIR = $(HOME)/.cache/rugby_ranking
CHECKPOINT_DIR = $(CACHE_DIR)/$(CHECKPOINT_NAME)
CHECKPOINT_ARCHIVE = $(CHECKPOINT_NAME).tar.gz

.PHONY: help mcmc mcmc-condor upload-release mcmc-and-upload clean

help:
	@echo "Rugby Ranking MCMC Training & Release Management"
	@echo ""
	@echo "Targets:"
	@echo "  make mcmc              - Train MCMC model locally (~4-8 hours)"
	@echo "  make mcmc-condor       - Generate HTCondor submit file and submit job"
	@echo "  make upload-release    - Upload checkpoint to GitHub Release"
	@echo "  make mcmc-and-upload   - Train and upload in one step"
	@echo "  make clean             - Remove old checkpoints and archives"
	@echo ""
	@echo "Configuration (can override):"
	@echo "  DATA_DIR=$(DATA_DIR)"
	@echo "  CHECKPOINT_NAME=$(CHECKPOINT_NAME)"
	@echo "  RELEASE_TAG=$(RELEASE_TAG)"
	@echo "  MCMC_DRAWS=$(MCMC_DRAWS)"
	@echo "  MCMC_CHAINS=$(MCMC_CHAINS)"
	@echo ""
	@echo "Examples:"
	@echo "  make mcmc MCMC_DRAWS=2000          - Train with 2000 draws"
	@echo "  make mcmc-condor LAST_SEASONS=3     - Submit HTCondor job for 3 seasons"
	@echo "  make upload-release CHECKPOINT_NAME=mcmc-2026-02"

# Train MCMC model locally
mcmc:
	@echo "=========================================="
	@echo "Training MCMC Model Locally"
	@echo "=========================================="
	@echo "Data directory: $(DATA_DIR)"
	@echo "Checkpoint name: $(CHECKPOINT_NAME)"
	@echo "Draws: $(MCMC_DRAWS), Chains: $(MCMC_CHAINS)"
	@echo "Last seasons: $(LAST_SEASONS)"
	@echo ""
	python train_model.py \
		--model $(MODEL_TYPE) \
		--method mcmc \
		--data-dir $(DATA_DIR) \
		--mcmc-draws $(MCMC_DRAWS) \
		--mcmc-chains $(MCMC_CHAINS) \
		--save-as $(CHECKPOINT_NAME) \
		--last-seasons $(LAST_SEASONS) \
		--verbose
	@echo ""
	@echo "✓ Training complete! Checkpoint saved to: $(CHECKPOINT_DIR)"
	@$(MAKE) compress

# Generate HTCondor submit file and submit job
mcmc-condor:
	@echo "=========================================="
	@echo "Generating HTCondor Submit File"
	@echo "=========================================="
	@mkdir -p condor_logs
	@cat > condor_mcmc_$(CHECKPOINT_NAME).sub <<EOF
	# HTCondor submit file for MCMC training
	# Generated: $(shell date)

	universe = vanilla
	executable = /usr/bin/python3
	arguments = train_model.py --model $(MODEL_TYPE) --method mcmc --data-dir $(DATA_DIR) --mcmc-draws $(MCMC_DRAWS) --mcmc-chains $(MCMC_CHAINS) --save-as $(CHECKPOINT_NAME) --last-seasons $(LAST_SEASONS) --verbose --checkpoint-every 100

	# Resource requirements
	request_cpus = $(MCMC_CHAINS)
	request_memory = 8GB
	request_disk = 5GB

	# Input/Output
	transfer_input_files = train_model.py, rugby_ranking/
	should_transfer_files = YES
	when_to_transfer_output = ON_EXIT

	# Logs
	log = condor_logs/mcmc_$(CHECKPOINT_NAME).log
	output = condor_logs/mcmc_$(CHECKPOINT_NAME).out
	error = condor_logs/mcmc_$(CHECKPOINT_NAME).err

	# Email notification (optional)
	# notify_user = your-email@example.com
	# notification = Complete

	# Requirements
	requirements = (OpSysMajorVer == 9)

	queue
	EOF
	@echo "✓ Submit file created: condor_mcmc_$(CHECKPOINT_NAME).sub"
	@echo ""
	@echo "To submit the job, run:"
	@echo "  condor_submit condor_mcmc_$(CHECKPOINT_NAME).sub"
	@echo ""
	@read -p "Submit job now? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		condor_submit condor_mcmc_$(CHECKPOINT_NAME).sub; \
		echo ""; \
		echo "Job submitted! Monitor with: condor_q"; \
		echo "View logs: tail -f condor_logs/mcmc_$(CHECKPOINT_NAME).log"; \
	fi

# Compress checkpoint for upload
compress:
	@echo "=========================================="
	@echo "Compressing Checkpoint"
	@echo "=========================================="
	@if [ ! -d "$(CHECKPOINT_DIR)" ]; then \
		echo "Error: Checkpoint directory not found: $(CHECKPOINT_DIR)"; \
		exit 1; \
	fi
	@cd $(CHECKPOINT_DIR) && \
		tar -czf $(CHECKPOINT_ARCHIVE) trace.nc metadata.pkl && \
		mv $(CHECKPOINT_ARCHIVE) $(shell pwd)/
	@echo "✓ Compressed checkpoint: $(CHECKPOINT_ARCHIVE)"
	@ls -lh $(CHECKPOINT_ARCHIVE)

# Upload checkpoint to GitHub Release
upload-release:
	@echo "=========================================="
	@echo "Uploading to GitHub Release"
	@echo "=========================================="
	@if [ ! -f "$(CHECKPOINT_ARCHIVE)" ]; then \
		echo "Error: Checkpoint archive not found: $(CHECKPOINT_ARCHIVE)"; \
		echo "Run 'make compress' first or 'make mcmc' to train and compress."; \
		exit 1; \
	fi
	@echo "Release tag: $(RELEASE_TAG)"
	@echo "Archive: $(CHECKPOINT_ARCHIVE)"
	@echo ""
	@# Check if release exists
	@if gh release view $(RELEASE_TAG) >/dev/null 2>&1; then \
		echo "Release $(RELEASE_TAG) already exists"; \
		echo "Uploading checkpoint to existing release..."; \
		gh release upload $(RELEASE_TAG) $(CHECKPOINT_ARCHIVE) --clobber; \
	else \
		echo "Creating new release $(RELEASE_TAG)..."; \
		gh release create $(RELEASE_TAG) \
			--title $(RELEASE_TITLE) \
			--notes "Monthly MCMC validation checkpoint\n\nTraining parameters:\n- Draws: $(MCMC_DRAWS)\n- Chains: $(MCMC_CHAINS)\n- Seasons: $(LAST_SEASONS)\n- Model: $(MODEL_TYPE)" \
			$(CHECKPOINT_ARCHIVE); \
	fi
	@echo ""
	@echo "✓ Upload complete!"
	@echo "View release: gh release view $(RELEASE_TAG) --web"

# Train MCMC and upload in one step
mcmc-and-upload: mcmc upload-release
	@echo ""
	@echo "=========================================="
	@echo "✓ MCMC Training & Upload Complete!"
	@echo "=========================================="
	@echo "Checkpoint: $(CHECKPOINT_NAME)"
	@echo "Release: $(RELEASE_TAG)"
	@echo ""
	@echo "Next weekly VI runs will automatically use this checkpoint."

# Clean up old checkpoints and archives
clean:
	@echo "Cleaning up old checkpoints and archives..."
	@rm -f mcmc-*.tar.gz
	@rm -f condor_mcmc_*.sub
	@echo "✓ Cleaned up archives and submit files"
	@echo ""
	@echo "To remove checkpoint cache:"
	@echo "  rm -rf $(CACHE_DIR)/mcmc-*"

# Verify checkpoint integrity
verify:
	@echo "=========================================="
	@echo "Verifying Checkpoint"
	@echo "=========================================="
	@if [ ! -f "$(CHECKPOINT_ARCHIVE)" ]; then \
		echo "Error: Checkpoint archive not found: $(CHECKPOINT_ARCHIVE)"; \
		exit 1; \
	fi
	@echo "Archive: $(CHECKPOINT_ARCHIVE)"
	@tar -tzf $(CHECKPOINT_ARCHIVE)
	@echo ""
	@python -c "import arviz as az; trace = az.from_netcdf('$(CHECKPOINT_DIR)/trace.nc'); print('Posterior shape:', trace.posterior.dims); print('✓ Checkpoint is valid')"

# Quick status check
status:
	@echo "=========================================="
	@echo "MCMC Training Status"
	@echo "=========================================="
	@echo "Cached checkpoints:"
	@ls -lh $(CACHE_DIR)/mcmc-* 2>/dev/null || echo "  (none)"
	@echo ""
	@echo "Local archives:"
	@ls -lh mcmc-*.tar.gz 2>/dev/null || echo "  (none)"
	@echo ""
	@echo "Recent releases:"
	@gh release list --limit 5 2>/dev/null || echo "  (gh not available or not authenticated)"
	@echo ""
	@echo "HTCondor jobs:"
	@condor_q 2>/dev/null || echo "  (condor not available)"
