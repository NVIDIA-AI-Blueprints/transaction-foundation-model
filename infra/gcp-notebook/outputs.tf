output "instance_name" {
  value = google_compute_instance.notebook.name
}

output "zone" {
  value = google_compute_instance.notebook.zone
}

output "service_account_email" {
  value = google_service_account.notebook.email
}

output "bucket" {
  value = google_storage_bucket.artifacts.name
}

output "remote_workspace" {
  value = "/mnt/tfm/workspace"
}

output "ssh_command" {
  value = "gcloud compute ssh ${google_compute_instance.notebook.name} --zone ${var.zone} --project ${var.project_id}"
}

