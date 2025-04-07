# Contributing to Spatter

Thank you for your interest in contributing to the Spatter open source project. We appreciate your interest and contributions. Please review these guidelines for the project so that you can most effectively engage with the developers and maintainers of the project. 

The license for the project is BSD-3 with some slight language additions from the main contributors at Los Alamos National Labs and Georgia Institute of Technology. 

## Types of Contributions Accepted

Spatter is a growing project and ecosystem, and there are several opportunities for contributions in the areas of new features, training and education, and general improvements to the codebase. 

### Contribution Process
In general, to contribute fixes or new code to Spatter, you would do the following:
1) Fork the Spatter codebase
2) Make your changes and run unit tests locally to see that changes do not break anything
3) Rebase on spatter-devel branch in the main repo, as needed. This is our "test" branch whereas main is the "stable" branch.
4) Open a PR and fill in the PR template with relevant information on your changes.
5) Request reviewers and discuss/update the PR until it is ready to be accepted.

### Reporting Bugs

**NOTE**: If you find a security vulnerability in the codebase, please do not open an issue. Email the primary maintainers or post a Discussion message noting you have something you'd like to report. This provides the development team an opportunity to respond to your findings and prepare a patch that can be released to mitigate any vulnerabilities.

Other general bugs can be reported using our [Bug Report Issue Template](https://github.com/hpcgarage/spatter/issues/new?assignees=&labels=bug&projects=&template=00-bug-report.yml&title=%F0%9F%90%9B+%5BBUG%5D+-+%3Ctitle%3E).

### Suggesting Features or Enhancements

If there is a feature that you would like to see in Spatter that doesn't currently exist, we also have a [feature request issue template](https://github.com/hpcgarage/spatter/issues/new?assignees=&labels=feature+request&projects=&template=01-feature-request.yml&title=%E2%9C%A8+%5BFEATURE+REQUEST%5D+-+%3Ctitle%3E) that you can use. This will create a new issue for further discussion with the development team. 

### New Spatter Backends

We have created a guide to developing new backends for Spatter [on our wiki here](https://github.com/hpcgarage/spatter/wiki/Adding-New-Backends-to-Spatter). If you would like to create a new backend for Spatter, you can create a new feature request, and the maintainers can link any 

## Code review process

We will aim to review your PR in 1-2 weeks with a goal of getting it merged within that time frame. If added changes are needed, we will follow up via the PR request and may make some suggestions to help your PR pass the CI unit tests and be able to be merged into the main line of the codebase. 

### Code style conventions

Spatter does not currently have strong checks for code formatting but please try to be consistent with the existing codebase with your changes. The `.clang-format` file in the top-level directory specifies the desired format for any commits. 

## Community Channels for Discussion

The primary discussion channels for this project are via [GitHub Issues](https://github.com/hpcgarage/spatter/issues) and the [GitHub Discussions space of this project](https://github.com/hpcgarage/spatter/discussions). You are welcome to post any general questions about Spatter to the Discussions channel or email the maintainers. 

### Code of Conduct for Discussions
The maintainers of the project aim to make participation in our community a harassment-free experience for everyone, regardless of age, body size, visible or invisible disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, caste, color, religion, or sexual identity and orientation.

We pledge to act and interact in ways that contribute to an open, welcoming, diverse, inclusive, and healthy community. For more details on our standard of conduct, please see the [Contributor Covenant Guidelines](https://www.contributor-covenant.org/version/2/1/code_of_conduct/), which we follow to ensure constructive and welcoming discussions and engagement.s
