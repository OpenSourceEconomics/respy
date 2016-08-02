provider "aws" {
  access_key = ""
  secret_key = ""
  region     = "us-east-1"
}

resource "aws_instance" "respy" {
  ami           = "ami-6457c773"
  instance_type = "t2.micro"
  key_name = "respy"
  security_groups=["terraform_example"]
    connection {
        user = "ubuntu"
        key_file = "/home/peisenha/.ssh/ec2-respy.pem"
    }
}

# Our default security group to access
# the instances over SSH and HTTP
resource "aws_security_group" "default" {
    name = "terraform_example"
    description = "Used in the terraform"

    # SSH access from anywhere
    ingress {
        from_port = 22
        to_port = 22
        protocol = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
    }

}
