# Declarative import: plan previews adoption before apply writes Terraform state.
# A missing variable causes a prompt/error, not an accidental second instance.
import {
  for_each = var.existing_instance_id == null ? {} : { api = var.existing_instance_id }
  to       = aws_instance.api
  id       = each.value
}

# Adopt the existing security group and its rules (requires create_security_group = true).
import {
  for_each = var.security_group_import == null ? {} : { api = var.security_group_import }
  to       = aws_security_group.api[0]
  id       = each.value.group_id
}

import {
  for_each = var.security_group_import == null ? {} : { api = var.security_group_import }
  to       = aws_vpc_security_group_ingress_rule.http[0]
  id       = each.value.http_rule_id
}

import {
  for_each = var.security_group_import == null ? {} : { api = var.security_group_import }
  to       = aws_vpc_security_group_egress_rule.outbound[0]
  id       = each.value.egress_rule_id
}
