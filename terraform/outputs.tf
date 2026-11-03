output "instance_id" {
  description = "Managed EC2 instance ID."
  value       = aws_instance.api.id
}

output "public_ip" {
  description = "Public IPv4 for GitHub Actions VM_HOST; may change after stop/start without an EIP."
  value       = aws_instance.api.public_ip
}

output "swagger_url" {
  description = "Available after the application is deployed and TCP 80 is reachable."
  value       = aws_instance.api.public_ip != "" ? "http://${aws_instance.api.public_ip}/docs" : null
}

output "security_group_ids" {
  value = aws_instance.api.vpc_security_group_ids
}
