mock_provider "aws" {
  # Random mock strings would fail ARN validation.
  mock_data "aws_partition" {
    defaults = { partition = "aws" }
  }
  mock_data "aws_caller_identity" {
    defaults = { account_id = "123456789012" }
  }
}

variables {
  aws_region                  = "eu-north-1"
  existing_instance_id        = null
  ami_id                      = "ami-0123456789abcdef0"
  instance_type               = "t3.micro"
  subnet_id                   = "subnet-0123456789abcdef0"
  key_name                    = "test-key"
  associate_public_ip_address = true
  root_volume_size            = 20
  root_volume_type            = "gp3"
  root_delete_on_termination  = true
  github_repository           = "octocat/example"
}

run "reuse_existing_security_groups" {
  command = plan

  variables {
    existing_security_group_ids = ["sg-0123456789abcdef0", "sg-0123456789abcdef1"]
  }

  assert {
    condition     = length(aws_security_group.api) == 0 && length(aws_vpc_security_group_ingress_rule.http) == 0
    error_message = "Adoption mode must not create or modify security group rules."
  }

  assert {
    condition     = aws_instance.api.vpc_security_group_ids == toset(["sg-0123456789abcdef0", "sg-0123456789abcdef1"])
    error_message = "Every existing security group must be retained."
  }

  assert {
    condition     = aws_instance.api.iam_instance_profile == "wind-api-instance"
    error_message = "The instance needs the SSM profile for keyless deployment."
  }
}

run "create_restricted_security_group" {
  command = plan

  variables {
    create_security_group = true
    ssh_allowed_cidrs     = ["203.0.113.10/32"]
  }

  assert {
    condition     = length(aws_security_group.api) == 1 && length(aws_vpc_security_group_ingress_rule.ssh) == 1
    error_message = "New group mode must create one dedicated group and the specified SSH rule."
  }

  assert {
    condition     = aws_vpc_security_group_ingress_rule.ssh["203.0.113.10/32"].from_port == 22 && aws_vpc_security_group_ingress_rule.http[0].from_port == 80
    error_message = "Ingress must match the existing HTTP/SSH deployment."
  }
}

run "managed_security_group_without_ssh" {
  command = plan

  variables {
    create_security_group = true
  }

  assert {
    condition     = length(aws_vpc_security_group_ingress_rule.ssh) == 0 && length(aws_vpc_security_group_ingress_rule.http) == 1
    error_message = "SSM deploys need no SSH rule; only HTTP should be open."
  }

  assert {
    condition     = aws_security_group.api[0].name_prefix == "wind-api-"
    error_message = "A new group without an explicit name should use the wind-api- prefix."
  }
}

run "reject_missing_security_groups" {
  command         = plan
  expect_failures = [aws_instance.api]
}

run "reject_world_open_ssh" {
  command = plan

  variables {
    create_security_group = true
    ssh_allowed_cidrs     = ["0.0.0.0/0"]
  }

  expect_failures = [var.ssh_allowed_cidrs]
}
