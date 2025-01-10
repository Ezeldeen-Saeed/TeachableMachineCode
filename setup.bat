:: Activate the virtual enviroment
echo Activating the virtual environment....
call "C:\Users\Ezeldeen Saeed\AppData\Local\python\mu\mu_venv-38-20250109-142517\Scripts\activate.bat"
if %errorlevel% neq 0 (
	echo Failed to activate the virtual environment. Exiting...
	pause
	exit /b 1
)

:: Upgrade pip
echo Upgrading pip
python.exe -m pip install --upgrade pip
if %errorlevel% neq 0 (
	echo Failed to upgrade pip. Exiting....
	call "C:\Users\Ezeldeen Saeed\AppData\Local\python\mu\mu_venv-38-20250109-142517\Scripts\deactivate.bat"
	pause
	exit /b 1
)

:: Install dependencies
echo Installing dependencies....
pip install tensorflow opencv-python numpy
if %errorlevel% neq 0 (
	echo Failed to install dependencies. Exiting....
	call "C:\Users\Ezeldeen Saeed\AppData\Local\python\mu\mu_venv-38-20250109-142517\Scripts\deactivate.bat"
	pause
	exit /b 1
)

:: Deactivate the virtual environment....
call "C:\Users\Ezeldeen Saeed\AppData\Local\python\mu\mu_venv-38-20250109-142517\Scripts\deactivate.bat"
if %errorlevel% neq 0 (
	echo Failed to deactivate the virtual environment. Exiting....
	pause
	exit /b 1
)

echo Setup complete!
exit
