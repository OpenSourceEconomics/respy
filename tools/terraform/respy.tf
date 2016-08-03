provider "aws" {
  access_key = ""
  secret_key = ""
  region     = "us-east-1"
}

resource "aws_instance" "respy" {
  ami           = "ami-6457c773"
  instance_type = "t2.micro"
  key_name = "respy"
  security_groups=["launch-wizard-2"]
    connection {
        user = "ubuntu"
        key_file = "/home/peisenha/.ssh/ec2-respy.pem"
    }
}
