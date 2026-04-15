"use client"

import { useEffect, useRef, useState } from "react"

interface OpenSAMAnnotatorLogoProps {
  size?: number
  showText?: boolean
}

export function OpenSAMAnnotatorLogo({ size = 200, showText = true }: OpenSAMAnnotatorLogoProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [rotation, setRotation] = useState(0)
  const currentRotationRef = useRef(0)
  const [fontLoaded, setFontLoaded] = useState(false)

  // Load Syne font dynamically
  useEffect(() => {
    if (!showText) return
    
    const link = document.createElement("link")
    link.href = "https://fonts.googleapis.com/css2?family=Syne:wght@800&display=swap"
    link.rel = "stylesheet"
    document.head.appendChild(link)
    
    link.onload = () => setFontLoaded(true)
    
    return () => {
      document.head.removeChild(link)
    }
  }, [showText])

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!containerRef.current) return

      const rect = containerRef.current.getBoundingClientRect()
      const centerX = rect.left + rect.width / 2
      const centerY = rect.top + rect.height / 2

      const dx = e.clientX - centerX
      const dy = e.clientY - centerY

      // Calculate angle from center to cursor (in degrees)
      const targetAngle = Math.atan2(dy, dx) * (180 / Math.PI)
      
      // The pupil is at ~(39, 45) with translate, center at (29, 29)
      // Vector from center to pupil: (10, 16), angle = atan2(16, 10) ≈ 58 degrees
      const adjustedTarget = targetAngle - 58

      // Get shortest rotation path to avoid spinning the long way around
      const current = currentRotationRef.current
      let delta = adjustedTarget - current

      // Normalize delta to [-180, 180] range for shortest path
      while (delta > 180) delta -= 360
      while (delta < -180) delta += 360

      const newRotation = current + delta
      currentRotationRef.current = newRotation
      setRotation(newRotation)
    }

    window.addEventListener("mousemove", handleMouseMove)
    return () => window.removeEventListener("mousemove", handleMouseMove)
  }, [])

  const textSize = size * 0.12

  return (
    <div
      ref={containerRef}
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: size * 0.08,
      }}
    >
      <svg
        width={size}
        height={size}
        viewBox="0 0 58 58"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        style={{
          cursor: "pointer",
          transform: `rotate(${rotation}deg)`,
          transition: "transform 0.15s ease-out",
          transformOrigin: "center center",
        }}
      >
        {/* Outer circle */}
        <circle cx="29" cy="29" r="25.5" stroke="white" />

        {/* Inner ellipse 3 (largest) */}
        <path
          d="M22.2726 18.647C32.6397 13.0889 44.4654 15.3848 48.7979 23.4657C53.1301 31.5466 48.4968 42.6658 38.1297 48.224C27.7628 53.7818 15.9382 51.4863 11.6057 43.4057C7.27336 35.3249 11.9057 24.2053 22.2726 18.647Z"
          stroke="white"
          transform="translate(3, 3)"
        />

        {/* Inner ellipse 2 */}
        <path
          d="M25.1001 23.9458C33.715 19.3271 43.5105 21.2486 47.0914 27.9277C50.6719 34.6067 46.8505 43.8279 38.2358 48.4465C29.6211 53.0651 19.826 51.1443 16.245 44.4655C12.6641 37.7865 16.4853 28.5645 25.1001 23.9458Z"
          stroke="white"
          transform="translate(3, 3)"
        />

        {/* Inner ellipse 1 (smallest) */}
        <path
          d="M30.0112 30.6274C35.9489 27.444 42.6433 28.7925 45.0759 33.3299C47.5085 37.8673 44.9266 44.1892 38.9888 47.3726C33.0511 50.556 26.3567 49.2075 23.9241 44.6701C21.4914 40.1327 24.0734 33.8108 30.0112 30.6274Z"
          stroke="white"
          transform="translate(3, 3)"
        />

        {/* Pupil */}
        <path
          d="M36.0275 41.1187C36.9513 40.6234 37.654 40.9779 37.822 41.2912C37.99 41.6046 37.8963 42.386 36.9725 42.8813C36.0487 43.3766 35.346 43.0222 35.178 42.7088C35.01 42.3954 35.1037 41.614 36.0275 41.1187Z"
          stroke="#00D4B4"
          strokeWidth="2"
          transform="translate(3, 3)"
        />
      </svg>

      {/* OpenSAMAnnotator text */}
      {showText && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            whiteSpace: "nowrap",
            fontFamily: fontLoaded ? "'Syne', sans-serif" : "system-ui, sans-serif",
            fontSize: textSize,
            fontWeight: 800,
            letterSpacing: "-0.025em",
          }}
        >
          <span style={{ color: "white" }}>Open</span>
          <span style={{ color: "#00D4B4" }}>SAM</span>
          <span style={{ color: "white" }}>Annotator</span>
        </div>
      )}
    </div>
  )
}
