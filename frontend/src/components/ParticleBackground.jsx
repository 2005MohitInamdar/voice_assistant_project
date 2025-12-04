// EdgeParticles.jsx
import React, { useRef, useEffect } from "react";
import "./ParticleBackground.css";

class EdgeParticle {
  constructor(canvas, edges, safeZone) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.size = Math.random() * 4 + 1;
    this.color = `hsl(${Math.random() * 360}, 100%, 70%)`;
    this.speed = Math.random() * 1 + 0.2;
    this.safeZone = safeZone; // { x, y, width, height }

    // Spawn along edge
    const edge = edges[Math.floor(Math.random() * edges.length)];
    this.spawn(edge);
  }

  spawn(edge) {
    switch (edge) {
      case "top":
        this.x = Math.random() * this.canvas.width;
        this.y = 0;
        this.vx = 0;
        this.vy = this.speed;
        break;
      case "bottom":
        this.x = Math.random() * this.canvas.width;
        this.y = this.canvas.height;
        this.vx = 0;
        this.vy = -this.speed;
        break;
      case "left":
        this.x = 0;
        this.y = Math.random() * this.canvas.height;
        this.vx = this.speed;
        this.vy = 0;
        break;
      case "right":
        this.x = this.canvas.width;
        this.y = Math.random() * this.canvas.height;
        this.vx = -this.speed;
        this.vy = 0;
        break;
    }
  }

  draw() {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
    ctx.fillStyle = this.color;
    ctx.fill();
    ctx.closePath();
  }

  update() {
    // Next position
    const nextX = this.x + this.vx;
    const nextY = this.y + this.vy;

    // Check if next position is inside safe zone
    const { x, y, width, height } = this.safeZone;
    if (
      nextX > x &&
      nextX < x + width &&
      nextY > y &&
      nextY < y + height
    ) {
      // If it would enter, reverse direction
      this.vx *= -1;
      this.vy *= -1;
    }

    this.x += this.vx;
    this.y += this.vy;

    // Wrap back to edge if out of canvas
    if (this.x < 0 || this.x > this.canvas.width || this.y < 0 || this.y > this.canvas.height) {
      this.spawn(["top", "bottom", "left", "right"][Math.floor(Math.random() * 4)]);
    }

    this.draw();
  }
}

const EdgeParticles = ({ count = 100, safeZone }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const edges = ["top", "bottom", "left", "right"];
    const particles = [];

    for (let i = 0; i < count; i++) {
      particles.push(new EdgeParticle(canvas, edges, safeZone));
    }

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      particles.forEach((p) => p.update());
      requestAnimationFrame(animate);
    };

    animate();

    const handleResize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [count, safeZone]);

  return <canvas ref={canvasRef} className="edgeParticlesCanvas" />;
};

export default EdgeParticles;
