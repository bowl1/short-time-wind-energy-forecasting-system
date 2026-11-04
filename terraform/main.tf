# Reuse the existing VPC/subnet. No EKS, NAT gateway, load balancer or EIP.
data "aws_subnet" "selected" {
  id = var.subnet_id
}

resource "aws_security_group" "api" {
  count = var.create_security_group ? 1 : 0
  # name/description force replacement, so an adopted group keeps its original values.
  name        = var.security_group_name
  name_prefix = var.security_group_name == null ? "wind-api-" : null
  description = var.security_group_description
  vpc_id      = data.aws_subnet.selected.vpc_id

  tags = { Name = "wind-api" }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_vpc_security_group_ingress_rule" "http" {
  count             = var.create_security_group ? 1 : 0
  security_group_id = aws_security_group.api[0].id
  description       = "Public API served by Docker on port 80"
  ip_protocol       = "tcp"
  from_port         = 80
  to_port           = 80
  cidr_ipv4         = "0.0.0.0/0"
}

resource "aws_vpc_security_group_ingress_rule" "ssh" {
  for_each          = var.create_security_group ? var.ssh_allowed_cidrs : toset([])
  security_group_id = aws_security_group.api[0].id
  description       = "Optional SSH from a trusted administrator"
  ip_protocol       = "tcp"
  from_port         = 22
  to_port           = 22
  cidr_ipv4         = each.value
}

resource "aws_vpc_security_group_egress_rule" "outbound" {
  count             = var.create_security_group ? 1 : 0
  security_group_id = aws_security_group.api[0].id
  description       = "Package installation, image downloads and external APIs"
  ip_protocol       = "-1"
  cidr_ipv4         = "0.0.0.0/0"
}

resource "aws_instance" "api" {
  ami                         = var.ami_id
  instance_type               = var.instance_type
  subnet_id                   = data.aws_subnet.selected.id
  key_name                    = var.key_name
  iam_instance_profile        = aws_iam_instance_profile.instance.name
  associate_public_ip_address = var.associate_public_ip_address
  vpc_security_group_ids      = var.create_security_group ? [aws_security_group.api[0].id] : var.existing_security_group_ids

  root_block_device {
    volume_size           = var.root_volume_size
    volume_type           = var.root_volume_type
    delete_on_termination = var.root_delete_on_termination
  }

  tags = var.instance_tags

  # Application installation/release remains in GitHub Actions; no provisioners.
  lifecycle {
    prevent_destroy = true

    precondition {
      condition     = var.create_security_group ? length(var.existing_security_group_ids) == 0 : length(var.existing_security_group_ids) > 0
      error_message = "Reuse existing security groups, OR let Terraform manage one; do not mix modes."
    }
  }
}
