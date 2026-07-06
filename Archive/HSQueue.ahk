; Fullscreen
; OCR disable pop-up window, preview

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
	MouseMove, 2715, 1903
	Send {LButton}
	Send {LButton}
	
	; Click to "Play" location
	MouseMove, 2715, 1903
	Send {LButton}
	
	ClipBoard =
	Sleep 50
	
	; SCAN BOTTOM LEFT FOR "CarlosDanger"
	MouseMove, 219, 1826
	Send {LWin down}
	Send {q down}
	Send {q up}
	Send {LWin up}
	MouseMove, 501, 1871
	Send {LButton}	
	
	Sleep 500
	
	clip = %clipboard%
	
	; msgbox %clip%
	
	if clip contains Ca,er,osD
	{		
		count = 0
		ClipBoard =
		
		; RANKED and CASUAL:  SCAN TOP LEFT FOR TARGET_CLASS
		MouseMove, 11, 141
		sleep 100
		Send {LWin down}
		Send {q down}
		Send {q up}
		Send {LWin up}
		MouseMove, 136, 186
		sleep 100
		Send {LButton}
		
		Sleep 500
		
		clip = %clipboard%
		
		; List of classes:
		; RO,OG,GU,UE
		; M5,MAG,MAC,AG,GE,EMMA
		; SH,HA,AM,AM
		; PR,RIE,pm,ES,ST,sr
		; WARL,ARL,RL,LO,OC,CK
		; WARR,ARR,RR,RIOR,IO,OR
		; DR,RU,UI,ID
		; PA,AL,LA,AD,DI
		; DE,EM
		; UN,TE,ER
		; EA,AT,TH,KN,NI,IG,GH,HT
		
		if clip contains M5,MAG,MAC,AG,GE,EMMA
		{
			SetTimer, Loop, Off
			;Send !{Enter}
					
			SoundBeep
			SoundBeep
			SoundBeep
		}
		else
		{
			concede()
		}		
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
	MouseMove, 1721, 751
	Send {esc}
	Sleep 500
	Send {LButton}	
	Send {LButton}
	Send {LButton}
	sleep 500
	
	; conirm concede
	MouseMove, 2715, 1903
	Send {LButton}
	Send {LButton}
	Send {LButton}
	sleep 500
	Send {LButton}
	Send {LButton}
	Send {LButton}
	
	; quest warning
	sleep 500
	MouseMove, 1477, 1306
	Send {LButton}
	Send {LButton}
	Send {LButton}
	
	
	
	 
	; click away results popup
	MouseMove, 2715, 1903
	SetTimer, ClickTimer, 250
	Sleep 5000
	SetTimer, ClickTimer, off
}