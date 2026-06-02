import React, { useEffect, useRef, useState } from 'react';

const PongGame = () => {
    const canvasRef = useRef(null);
    const [paddleY, setPaddleY] = useState(250);
    const ballPos = useRef({ x: 400, y: 300 });
    const ballVel = useRef({ x: 5, y: 5 });
    const paddleHeight = 100;
    const paddleWidth = 20;

    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        let animationFrameId;

        const update = () => {
            // Ball movement
            ballPos.current.x += ballVel.current.x;
            ballPos.current.y += ballVel.current.y;

            // Wall collisions
            if (ballPos.current.y <= 0 || ballPos.current.y >= 600) {
                ballVel.current.y *= -1;
            }
            if (ballPos.current.x >= 800) {
                ballVel.current.x *= -1;
            }

            // Paddle collision (Left side)
            if (
                ballPos.current.x <= paddleWidth + 10 &&
                ballPos.current.y > paddleY &&
                ballPos.current.y < paddleY + paddleHeight
            ) {
                ballVel.current.x = Math.abs(ballVel.current.x); // Bounce off
            }

            // Reset if missed
            if (ballPos.current.x < 0) {
                ballPos.current = { x: 400, y: 300 };
                ballVel.current = { x: 5, y: 5 };
            }

            // Draw
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, 800, 600);

            ctx.fillStyle = 'white';
            // Paddle
            ctx.fillRect(10, paddleY, paddleWidth, paddleHeight);
            // Ball
            ctx.beginPath();
            ctx.arc(ballPos.current.x, ballPos.current.y, 10, 0, Math.PI * 2);
            ctx.fill();

            animationFrameId = window.requestAnimationFrame(update);
        };

        animationFrameId = window.requestAnimationFrame(update);
        return () => window.cancelAnimationFrame(animationFrameId);
    }, [paddleY]);

    const handleMouseMove = (e) => {
        const rect = canvasRef.current.getBoundingClientRect();
        const y = e.clientY - rect.top - paddleHeight / 2;
        setPaddleY(Math.max(0, Math.min(y, 600 - paddleHeight)));
    };

    const handleTouchMove = (e) => {
        const rect = canvasRef.current.getBoundingClientRect();
        const touch = e.touches[0];
        const y = touch.clientY - rect.top - paddleHeight / 2;
        setPaddleY(Math.max(0, Math.min(y, 600 - paddleHeight)));
    };

    return (
        <div style={{ 
            width: '100vw', 
            height: '100vh', 
            background: '#222', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            overflow: 'hidden',
            touchAction: 'none' // Prevent scrolling on Smart TV
        }}>
            <canvas
                ref={canvasRef}
                width={800}
                height={600}
                onMouseMove={handleMouseMove}
                onTouchMove={handleTouchMove}
                style={{ 
                    border: '5px solid white', 
                    cursor: 'none',
                    width: '90%', 
                    height: 'auto',
                    maxWidth: '800px'
                }}
            />
            <div style={{ 
                position: 'fixed', 
                top: 20, 
                color: 'white', 
                fontSize: '24px', 
                fontFamily: 'sans-serif' 
            }}>
                Static CT Pong Demo - Drag to Play
            </div>
        </div>
    );
};

export default PongGame;
