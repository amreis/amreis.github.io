---
title: Nix-Darwin 'failed to allocate' error on MacOS 27
date: 2026-09-15 12:00:00 -0300
description: Solving 'failed to allocate' error on nix-darwin after MacOS 27 update
tags: software-development nix nix-darwin
categories: misc
type: dev
layout: post
---

If you've recently updated your MacOS to version 27 Golden Gate and have started running into issues with `nix-darwin`, you've come to the right place.

### The issue

```sh
setting up /Applications/Nix Apps...
setting up pam...
applying patches...
setting up /etc...
user defaults...
restarting Dock...
setting up launchd services...
configuring networking...
configuring application firewall...
configuring power...
setting up /Library/Fonts/Nix Fonts...
setting nvram variables...
Homebrew bundle...
Using visual-studio-code
Using (...)
`brew bundle` complete! 10 Brewfile dependencies now installed.
failed to allocate 1048576 bytes at 0x300100000
(hint: Try "ulimit -a"; maybe you should increase memory limits.)
```

Those last two lines show an *error* that prevents `sudo darwin-rebuild switch --flake .` from running correctly.

After searching for a while, I saw that people tracked down the error to a version of SBCL (see [here](https://github.com/hraban/mac-app-util/issues/20), for example). It took me _longer_ to realize that the issue was coming from [mac-app-util](https://github.com/hraban/mac-app-util).

The fix is simple: in your Flake, tell `mac-app-util` to use the same `nixpkgs` input as the rest of your flake. That is, the fix is to change this:

```nix
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    nix-darwin = {
      url = "github:nix-darwin/nix-darwin/master";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    mac-app-util = {
        url = "github:hraban/mac-app-util";
    };
  };
```
to this:

```nix
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    nix-darwin = {
      url = "github:nix-darwin/nix-darwin/master";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    mac-app-util = {
        url = "github:hraban/mac-app-util";
        # !!!
        inputs.nixpkgs.follows = "nixpkgs";  # temporarily override nixpkgs url to get SBCL v2.6.6
    };
  };
```

That should do it ☺️.