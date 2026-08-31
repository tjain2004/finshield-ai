@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title FinShield AI - Push deploy folder to GitHub

echo ================================================================
echo    FinShield AI  -  push this deploy folder to GitHub
echo ================================================================
echo.

REM ---------------------------------------------------------------
REM  0 - is git installed?
REM ---------------------------------------------------------------
git --version >nul 2>&1
if errorlevel 1 goto NO_GIT
for /f "tokens=*" %%V in ('git --version') do echo    %%V
echo.

REM ---------------------------------------------------------------
REM  1 - who is committing?
REM ---------------------------------------------------------------
set GNAME=
set GMAIL=
for /f "tokens=*" %%A in ('git config user.name 2^>nul') do set GNAME=%%A
for /f "tokens=*" %%A in ('git config user.email 2^>nul') do set GMAIL=%%A

if "!GNAME!"=="" goto ASK_NAME
if "!GMAIL!"=="" goto ASK_NAME
echo [1/7] committing as  !GNAME!  ^<!GMAIL!^>
goto ASK_REPO

:ASK_NAME
echo [1/7] Git does not know who you are yet.
echo.
set /p GNAME="      Your name: "
set /p GMAIL="      Your GitHub email: "
if "!GNAME!"=="" goto BAD_INPUT
if "!GMAIL!"=="" goto BAD_INPUT
git config --global user.name "!GNAME!"
git config --global user.email "!GMAIL!"
echo       saved.
echo.

REM ---------------------------------------------------------------
REM  2 - which repository?
REM ---------------------------------------------------------------
:ASK_REPO
echo.
echo [2/7] Create a NEW EMPTY public repo on GitHub first, then paste its URL.
echo.
echo       Press ENTER to accept the default shown below, or paste your own.
echo.
echo       Default:  https://github.com/tjain2004/finshield-dashboard.git
echo.
set /p REPO="      Repository URL: "
if "!REPO!"=="" set REPO=https://github.com/tjain2004/finshield-dashboard.git
echo.

REM ---------------------------------------------------------------
REM  3 - init
REM ---------------------------------------------------------------
if exist ".git" goto HAVE_REPO
echo [3/7] git init
git init >nul
if errorlevel 1 goto GIT_FAIL
goto STAGE

:HAVE_REPO
echo [3/7] a git repo already exists in this folder - reusing it

:STAGE
REM ---------------------------------------------------------------
REM  4 - stage + commit
REM ---------------------------------------------------------------
echo [4/7] staging files...
git add -A
if errorlevel 1 goto GIT_FAIL
for /f %%N in ('git diff --cached --name-only ^| find /c /v ""') do set NFILES=%%N
echo       !NFILES! files staged

echo [5/7] committing...
git commit -m "FinShield AI dashboard - cloud build" >nul 2>&1
if errorlevel 1 echo       nothing new to commit - continuing

REM ---------------------------------------------------------------
REM  5 - branch + remote
REM ---------------------------------------------------------------
echo [6/7] setting branch to main and remote to your repo
git branch -M main
git remote remove origin >nul 2>&1
git remote add origin "!REPO!"
if errorlevel 1 goto GIT_FAIL

REM ---------------------------------------------------------------
REM  6 - push
REM ---------------------------------------------------------------
echo [7/7] pushing to GitHub...
echo.
echo       A browser or a sign-in box may pop up. Sign in to GitHub
echo       and allow it. This only happens the first time.
echo.
git push -u origin main
if errorlevel 1 goto PUSH_FAIL

echo.
echo ================================================================
echo    SUCCESS - your code is on GitHub.
echo.
echo    !REPO!
echo.
echo    Next: go to render.com, sign in with GitHub, then
echo    New  -  Web Service  -  pick this repo  -  Instance Type FREE
echo ================================================================
echo.
pause
exit /b 0

REM ===============================================================
:NO_GIT
echo    Git is not installed on this computer.
echo.
echo    Download it from:  https://git-scm.com/download/win
echo    Click Next through every screen, then close this window,
echo    reopen the folder and double-click this file again.
echo.
pause
exit /b 1

:BAD_INPUT
echo.
echo    Nothing entered - stopping so nothing gets half-done.
echo.
pause
exit /b 1

:GIT_FAIL
echo.
echo    A git command failed. Read the message above this line.
echo.
pause
exit /b 1

:PUSH_FAIL
echo.
echo ================================================================
echo    THE PUSH FAILED. The usual causes:
echo.
echo    "Repository not found"
echo       The URL is wrong, or the repo is private. Free Render
echo       needs a PUBLIC repo. Check the URL and run this again.
echo.
echo    "Authentication failed"
echo       Wrong GitHub login. Run this again and sign in properly.
echo.
echo    "this exceeds GitHub's file size limit"
echo       You are pushing the wrong folder. This file must live in
echo       the deploy folder, not the main FinShieldAI folder.
echo.
echo    "Updates were rejected"
echo       The repo you made was not empty. Make a brand new one
echo       with no README and no gitignore, then run this again.
echo ================================================================
echo.
pause
exit /b 1
