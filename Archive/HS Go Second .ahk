; Fullscreen
; OCR disable pop-up window, preview
; HS Deck Track ---> Advanced ---> Player ---> Scaling 50, Opponent ---> Scaling 50, Secret scaling 50
; Must fix starting hand counter location on deck change (deck list gets longer/shorter)

#SingleInstance force
StringCaseSense Off
CoordMode, Mouse, Window

`::
	exitApp
return

ClickTimer:
	Send {LButton}
return

; Reset bad match-up
f10::
	SoundBeep
	;Send !{Enter}
	
	Sleep 1000
	
	concede()
	
	Send {f11}
return

f11::	
	SetTimer, Loop, 1
	SoundBeep
return 


f12::	
	SoundBeep
	exitApp
	
return 

count = 0

Loop:
	; Click off options menu if we missed clicking concede
	MouseMove, 2707, 1895
	Send {LButton}
	Send {LButton}
	
	; Click to "Play" location
	MouseMove, 2707, 1895
	Send {LButton}
	
	ClipBoard =
	Sleep 50
	
	; Scan for starting hand size on deck tracker
	MouseMove, 3272, 900
	Send {LWin down}
	Send {q down}
	Send {q up}
	Send {LWin up}
	MouseMove, 3317, 926
	Send {LButton}	
	
	
	Sleep 500
	
	clip = %clipboard%
	
	; msgbox %clip%
	
	if clip contains 5
	{	
		SetTimer, Loop, Off
		SoundBeep
		SoundBeep
		SoundBeep			
	}
	else
	{
		sleep 5000
		concede()
	}
	
	if(count == 300)
	{
		SetTimer, Loop, Off
		count = 0			
		
		SoundBeep
		SoundBeep
		SoundBeep
		SoundBeep
		SoundBeep
		SoundBeep
		SoundBeep
		SoundBeep
		SoundBeep
		SoundBeep
	}
	
	count++
	
return

concede()
{
	; Concede
	MouseMove, 1726, 750
	Send {esc}
	Sleep 1000
	Send {LButton}
	Send {LButton}
	Send {LButton}
	 
	; click away results popup
	MouseMove, 2707, 1895
	SetTimer, ClickTimer, 250
	Sleep 10000
	SetTimer, ClickTimer, off
}