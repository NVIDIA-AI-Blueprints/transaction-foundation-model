variable "project_id" {
  description = "GCP project ID."
  type        = string
}

variable "region" {
  description = "GCP region for regional resources."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone for the notebook VM and persistent disk."
  type        = string
  default     = "us-central1-f"
}

variable "name" {
  description = "Base name for the notebook VM resources."
  type        = string
  default     = "tfm-gpu-notebook"
}

variable "machine_type" {
  description = "GPU machine type. Default is A100 40GB because this project has A100 quota. Use a2-ultragpu-1g for A100 80GB only after quota approval, or g2-standard-24 for 2x L4."
  type        = string
  default     = "a2-highgpu-1g"
}

variable "boot_disk_size_gb" {
  description = "Boot disk size in GB."
  type        = number
  default     = 200
}

variable "data_disk_size_gb" {
  description = "Persistent data disk size in GB."
  type        = number
  default     = 1024
}

variable "data_disk_type" {
  description = "Persistent data disk type."
  type        = string
  default     = "pd-balanced"
}

variable "bucket_name" {
  description = "GCS bucket for datasets, checkpoints, and artifacts. Defaults to <project_id>-tfm-gpu-artifacts."
  type        = string
  default     = null
}

variable "bucket_location" {
  description = "GCS bucket location."
  type        = string
  default     = "US"
}

variable "network" {
  description = "VPC network self-link or name."
  type        = string
  default     = "default"
}

variable "subnetwork" {
  description = "Optional subnetwork self-link or name."
  type        = string
  default     = null
}

variable "enable_external_ip" {
  description = "Attach an external IP for SSH. Jupyter still binds only to localhost on the VM."
  type        = bool
  default     = true
}

variable "use_spot" {
  description = "Use Spot provisioning. Keep false for interactive notebooks unless interruptions are acceptable."
  type        = bool
  default     = false
}

variable "auto_shutdown_minutes" {
  description = "Schedule an OS shutdown after startup to cap runaway GPU cost. Set 0 to disable."
  type        = number
  default     = 720
}

variable "notebook_image" {
  description = "Default container image recorded on the VM for helper scripts."
  type        = string
  default     = "nvcr.io/nvidia/nemo:25.09.01"
}

variable "guest_accelerator_type" {
  description = "Optional accelerator type for N1-style VMs. Leave null for fixed-GPU machine types like A2/G2."
  type        = string
  default     = null
}

variable "guest_accelerator_count" {
  description = "Optional accelerator count when guest_accelerator_type is set."
  type        = number
  default     = 0
}
