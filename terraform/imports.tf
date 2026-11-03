# Declarative import: plan previews adoption before apply writes Terraform state.
# A missing variable causes a prompt/error, not an accidental second instance.
import {
  for_each = var.existing_instance_id == null ? {} : { api = var.existing_instance_id }
  to       = aws_instance.api
  id       = each.value
}
