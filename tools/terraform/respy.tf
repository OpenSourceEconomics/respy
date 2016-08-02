provider "aws" {
  access_key = ""
  secret_key = ""
  region     = "us-east-1"
}

resource "aws_instance" "respy" {
  ami           = "ami-4a8d1d5d"
  instance_type = "t2.micro"
}
