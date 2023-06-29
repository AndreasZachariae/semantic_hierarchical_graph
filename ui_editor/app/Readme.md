# Petra Map Editor Application

## Description

Petra Map Editor is designed to take in a map and display it to the user. The user can edit paths and areas and add constraints to those areas and save the map afterwards.

## Installation

This project uses Tauri to create a platform dependent distribution. To make sure this works follow the instructions on the [Tauri Installation Guide](https://tauri.app/v1/guides/getting-started/prerequisites).
Afterwards you can install the frontend dependencies using
```bash
npm install
```

## Run development build:
```bash
npm run tauri dev
```
This will start the frontend server and create a window showing the application.

## Build for production:
```bash
npm run tauri build
```

Note that this will package the application for the operating system you are running this command on, since Tauri does not yet support cross-compilation.