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
  description = "Manage the instance's security group in Terraform: create a new one, or adopt one via security_group_import."
  type        = bool
  default     = false
}

variable "ssh_allowed_cidrs" {
  description = "Optional trusted IPv4 CIDRs for SSH in a managed group. Deploys use SSM, so empty is recommended."
  type        = set(string)
  default     = []

  validation {
    condition = alltrue([
      for cidr in var.ssh_allowed_cidrs : can(cidrnetmask(cidr)) && cidr != "0.0.0.0/0"
    ])
    error_message = "Provide IPv4 CIDRs for trusted SSH clients; unrestricted 0.0.0.0/0 is not supported."
  }
}

variable "github_repository" {
  description = "GitHub owner/repo allowed to deploy via OIDC, e.g. octocat/example. Must be the current repo name, not a redirect."
  type        = string

  validation {
    condition     = can(regex("^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", var.github_repository))
    error_message = "Use the owner/repo form."
  }
}

variable "github_deploy_branch" {
  description = "Only workflows running on this branch can assume the deploy role."
  type        = string
  default     = "master"
}

variable "security_group_name" {
  description = "Exact name of the managed group. Required when adopting (renaming forces replacement); null uses the wind-api- prefix."
  type        = string
  nullable    = true
  default     = null
}

variable "security_group_description" {
  description = "Description of the managed group. Must match the existing group when adopting (changes force replacement)."
  type        = string
  default     = "HTTP API for wind power forecasts; deploys use SSM"
}

variable "security_group_import" {
  description = "IDs of an existing group and its HTTP/egress rules to adopt into the managed group; null when creating a new one."
  type = object({
    group_id       = string
    http_rule_id   = string
    egress_rule_id = string
  })
  nullable = true
  default  = null
}
