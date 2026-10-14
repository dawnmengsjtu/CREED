on run argv
	set inPath to item 1 of argv
	set outPath to item 2 of argv
	tell application "Pages Creator Studio"
		set theDoc to open (POSIX file inPath as alias)
		delay 3
		export theDoc to (POSIX file outPath) as PDF
		close theDoc saving no
	end tell
end run
