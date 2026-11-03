variable "aws_region" {
  description = "Region containing the existing EC2 instance and subnet."
  type        = string
}

variable "existing_instance_id" {
  description = "EC2 ID to import. Explicitly set null only when intentionally creating a new, billable instance."
  type        = string
  nullable    = true

  validation {
    condition     = var.existing_instance_id == null ? true : can(regex("^i-[0-9a-f]+$", var.existing_instance_id))
    error_message = "Use an EC2 instance ID (i-...) or explicitly set null for a new instance."
  }
}

variable "ami_id" {
  description = "Exact AMI ID of the existing instance, or a pinned Ubuntu AMI for a new instance."
  type        = string
}

variable "instance_type" {
  description = "Match the existing instance type when importing; changes can stop/restart EC2."
  type        = string
}

variable "subnet_id" {
  description = "Existing subnet. A new public deployment needs an Internet Gateway route and public IPv4."
  type        = string
}

variable "key_name" {
  description = "Existing EC2 key pair name; never put a private key in Terraform. Match the instance when importing."
  type        = string
  nullable    = true
}

variable "associate_public_ip_address" {
  description = "Match the existing instance on import. Public IPv4 can incur AWS charges."
  type        = bool
}

variable "root_volume_size" {
  description = "Root EBS volume size in GiB; match the existing disk when importing."
  type        = number

  validation {
    condition     = var.root_volume_size >= 8 && floor(var.root_volume_size) == var.root_volume_size
    error_message = "Root volume size must be an integer of at least 8 GiB."
  }
}

variable "root_volume_type" {
  description = "Root EBS volume type, e.g. gp2 or gp3; match the existing disk when importing."
  type        = string
}

variable "root_delete_on_termination" {
  description = "Match the existing EBS DeleteOnTermination setting when importing."
  type        = bool
}

variable "instance_tags" {
  description = "Full set of user-defined instance tags; copy existing tags to avoid unintended removal."
  type        = map(string)
  default     = {}
}

variable "existing_security_group_ids" {
  description = "Existing security groups to retain unchanged. Include ALL groups attached to the instance."
  type        = set(string)
  default     = []
}

variable "create_security_group" {
  description = "Opt in to a dedicated managed security group instead of reusing existing ones."
  type        = bool
  default     = false
}

variable "ssh_allowed_cidrs" {
  description = "Trusted IPv4 CIDRs for SSH when creating a group. Must also allow the deployment runner's egress IP."
  type        = set(string)
  default     = []

  validation {
    condition = alltrue([
      for cidr in var.ssh_allowed_cidrs : can(cidrnetmask(cidr)) && cidr != "0.0.0.0/0"
    ])
    error_message = "Provide IPv4 CIDRs for trusted SSH clients; unrestricted 0.0.0.0/0 is not supported."
  }
}
