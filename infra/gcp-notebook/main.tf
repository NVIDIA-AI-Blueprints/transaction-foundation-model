locals {
  bucket_name     = coalesce(var.bucket_name, "${var.project_id}-tfm-gpu-artifacts")
  data_disk_name  = "${var.name}-data"
  service_account = "${var.name}-sa"
  startup_script = templatefile("${path.module}/startup.sh", {
    data_disk_name        = local.data_disk_name
    auto_shutdown_minutes = var.auto_shutdown_minutes
    notebook_image        = var.notebook_image
    bucket_name           = local.bucket_name
  })
}

resource "google_project_service" "required" {
  for_each = toset([
    "compute.googleapis.com",
    "iam.googleapis.com",
    "storage.googleapis.com",
    "bigquery.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
  ])

  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

resource "google_service_account" "notebook" {
  account_id   = local.service_account
  display_name = "Transaction FM GPU notebook runtime"

  depends_on = [google_project_service.required]
}

resource "google_project_iam_member" "bigquery_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.notebook.email}"
}

resource "google_project_iam_member" "bigquery_data_viewer" {
  project = var.project_id
  role    = "roles/bigquery.dataViewer"
  member  = "serviceAccount:${google_service_account.notebook.email}"
}

resource "google_project_iam_member" "bigquery_read_session_user" {
  project = var.project_id
  role    = "roles/bigquery.readSessionUser"
  member  = "serviceAccount:${google_service_account.notebook.email}"
}

resource "google_project_iam_member" "logging_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.notebook.email}"
}

resource "google_project_iam_member" "monitoring_metric_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.notebook.email}"
}

resource "google_storage_bucket" "artifacts" {
  name                        = local.bucket_name
  location                    = var.bucket_location
  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  depends_on = [google_project_service.required]
}

resource "google_storage_bucket_iam_member" "notebook_bucket_admin" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.notebook.email}"
}

resource "google_compute_disk" "data" {
  name = local.data_disk_name
  type = var.data_disk_type
  zone = var.zone
  size = var.data_disk_size_gb

  depends_on = [google_project_service.required]
}

resource "google_compute_instance" "notebook" {
  name                      = var.name
  zone                      = var.zone
  machine_type              = var.machine_type
  allow_stopping_for_update = true

  boot_disk {
    initialize_params {
      image = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts"
      size  = var.boot_disk_size_gb
      type  = "pd-balanced"
    }
  }

  attached_disk {
    source      = google_compute_disk.data.id
    device_name = local.data_disk_name
    mode        = "READ_WRITE"
  }

  dynamic "guest_accelerator" {
    for_each = var.guest_accelerator_type == null ? [] : [1]

    content {
      type  = var.guest_accelerator_type
      count = var.guest_accelerator_count
    }
  }

  network_interface {
    network    = var.network
    subnetwork = var.subnetwork

    dynamic "access_config" {
      for_each = var.enable_external_ip ? [1] : []
      content {}
    }
  }

  scheduling {
    automatic_restart   = var.use_spot ? false : true
    on_host_maintenance = "TERMINATE"
    provisioning_model  = var.use_spot ? "SPOT" : "STANDARD"
    preemptible         = var.use_spot
  }

  service_account {
    email  = google_service_account.notebook.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    startup-script       = local.startup_script
    tfm-notebook-image   = var.notebook_image
    tfm-artifacts-bucket = google_storage_bucket.artifacts.name
    tfm-remote-workspace = "/mnt/tfm/workspace"
  }

  tags = ["tfm-gpu-notebook"]

  depends_on = [
    google_compute_disk.data,
    google_storage_bucket_iam_member.notebook_bucket_admin,
    google_project_iam_member.bigquery_job_user,
    google_project_iam_member.bigquery_data_viewer,
    google_project_iam_member.bigquery_read_session_user,
    google_project_iam_member.logging_log_writer,
    google_project_iam_member.monitoring_metric_writer,
  ]
}
